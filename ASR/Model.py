from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from torch.nn.utils.rnn import pad_sequence
import re
from transformers import AutoTokenizer
import speechbrain as sb
from speechbrain.utils.checkpoints import Checkpointer
from ASR.prepare import load_tedlium_dataset
import torch
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.dataio.dataset import DynamicItemDataset
import librosa
from torch.nn.functional import log_softmax

class ASR(sb.Brain):
    def __init__(self, modules=None, hparams=None, run_opts=None):
        if modules is None:
            modules = {}

        # Loading Wav2Vec2 Models and Processors
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.wav2vec2 = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

        modules["wav2vec2"] = self.wav2vec2
        self.checkpointer = Checkpointer(checkpoints_dir="./checkpoints", recoverables={
            "model": modules["wav2vec2"]
        })
        super().__init__(modules=modules, hparams=hparams, run_opts=run_opts)

    def compute_forward(self, batch, stage):
        # Getting Audio and Tags
        audios, _, _ = batch
        # print("Audio batch size:", audios.size(0))

        input_values = self.processor(
            audios.cpu().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        ).input_values.to(self.device)

        # forward propagation
        outputs = self.modules.wav2vec2(input_values)

        logits = outputs.logits

        return logits

    def compute_objectives(self, predictions, batch, stage):

        audios_padded, _, encoded_targets = batch

        # The dimensions of the transposed predictions are (T, N, C)
        predictions = predictions.permute(1, 0, 2)

        # Converting predictions to log probabilities
        log_probs = torch.nn.functional.log_softmax(predictions, dim=-1)

        input_lengths = torch.full(
            size=(predictions.size(1),),
            fill_value=predictions.size(0),
            dtype=torch.long
        )

        target_lengths = torch.tensor([len(target) for target in encoded_targets], dtype=torch.long)

        if target_lengths.size(0) != predictions.size(1):
            raise ValueError(
                f"Mismatch: target_lengths ({target_lengths.size(0)}) != batch_size ({predictions.size(1)})")

        assert all(t > 0 for t in target_lengths), "Target sequence contains empty elements!"

        # Calculating CTC Loss
        ctc_loss = torch.nn.CTCLoss(blank=self.processor.tokenizer.pad_token_id, zero_infinity=True)
        loss = ctc_loss(log_probs, encoded_targets, input_lengths, target_lengths)

        if loss.item() < -1e3:
            raise ValueError(f"Unexpected negative loss: {loss.item()}")

        return loss

    def on_stage_start(self, stage, epoch=None):
        if stage == sb.Stage.TRAIN:
            print(f"Training started for epoch {epoch}")
            # Make sure the model is in training mode
            self.modules.wav2vec2.train()
        elif stage == sb.Stage.VALID:
            print("Validation started")
            # Ensure that the model is in evaluation mode
            self.modules.wav2vec2.eval()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == sb.Stage.TRAIN:
            print(f"Epoch {epoch} Training Loss: {stage_loss}")
            # Using Checkpointer to Save Models

        elif stage == sb.Stage.VALID:
            print(f"Validation Loss: {stage_loss}")

def transcribe_audio(audio_path, asr_model, processor):
    """
    对输入音频文件进行语音转录。

    Args:
        audio_path (str): 音频文件路径。
        asr_model (ASR): 已加载的 ASR 模型实例。
        processor (Wav2Vec2Processor): 用于预处理音频和解码结果的 processor 实例。

    Returns:
        str: 转录结果。
    """
    # 加载音频并调整采样率到 16kHz
    audio, sr = librosa.load(audio_path, sr=16000)

    # 使用 processor 将音频转换为输入张量
    input_values = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    ).input_values.to(asr_model.device)

    # 模型推理
    asr_model.modules.wav2vec2.eval()  # 切换模型到评估模式
    with torch.no_grad():
        logits = asr_model.modules.wav2vec2(input_values).logits

    # 转换为对数概率
    log_probs = log_softmax(logits, dim=-1)

    # 贪婪解码：选取每一时间步概率最大的字符
    predicted_ids = torch.argmax(log_probs, dim=-1)

    # 使用 processor 解码预测结果
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription