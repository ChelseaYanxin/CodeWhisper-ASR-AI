from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from torch.nn.utils.rnn import pad_sequence
import re
from transformers import AutoTokenizer
import speechbrain as sb
from speechbrain.utils.checkpoints import Checkpointer 
from ASR.prepare import load_tedlium_dataset
from ASR.Model import ASR
import torch
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.dataio.dataset import DynamicItemDataset
import librosa
from torch.nn.functional import log_softmax
# Load data sets
tedlium = load_tedlium_dataset()
tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")

class AudioDataset(Dataset):
    def __init__(self, audio_samples, labels):
        self.audio_samples = audio_samples
        self.labels = labels

    def __len__(self):
        return len(self.audio_samples)

    def __getitem__(self, idx):
        audio_data, sample_rate = self.audio_samples[idx]
        label = self.labels[idx]
        return torch.tensor(audio_data, dtype=torch.float32), label
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")



def clean_text(text):
    # Removing content and special symbols in curly brackets
    text = re.sub(r"\{.*?\}", "", text)  
    text = re.sub(r"<.*?>", "", text)   
    text = re.sub(r"\(.*?\)", "", text)  
    return text.strip() 
def collate_fn(batch):
    audios = [item['processed_audio'] for item in batch]
    sample_rates = [item['sample_rate'] for item in batch]
    texts = [clean_text(item['processed_text']) for item in batch]


    # Fill audio to the same length
    audios_padded = pad_sequence(audios, batch_first=True)

    # Encoding Text with Tokenizer
    encoded_targets = tokenizer(
        texts,
        padding=True,
        return_tensors="pt",
        truncation=True
    ).input_ids

    return audios_padded, sample_rates,encoded_targets






def extract_data_from_dataset(dataset_split, subset_percent):
    audio_samples = []
    labels = []

    subset_size = int(len(dataset_split) * (subset_percent / 100.0))

    # Extracting audio samples and tags from datasets
    for i, item in enumerate(dataset_split):
        if i >= subset_size:
            break
        audio_data = item["audio"]["array"]  # Extracting audio arrays
        sample_rate = item["audio"]["sampling_rate"]  # Extraction Sampling Rate
        transcription = item["text"]  # Extract text labels

        audio_samples.append((audio_data, sample_rate))
        labels.append(transcription)

    return audio_samples, labels

# Functions to prepare a data dictionary
def prepare_data_dict(audio_samples, transcriptions):
    return {f"item_{i}": {"audio_samples": audio,  "labels": transcription}
            for i, (audio, transcription) in enumerate(zip(audio_samples, transcriptions))}


train_audio_samples, train_transcriptions = extract_data_from_dataset(tedlium["train"], subset_percent=0.1)
valid_audio_samples, valid_transcriptions = extract_data_from_dataset(tedlium["validation"], subset_percent=5)

# Creating a Data Dictionary
train_data_dict = prepare_data_dict(train_audio_samples, train_transcriptions)
valid_data_dict = prepare_data_dict(valid_audio_samples, valid_transcriptions)

# Creating a Dynamic Dataset Example
train_data = DynamicItemDataset(train_data_dict)
valid_data = DynamicItemDataset(valid_data_dict)

def audio_pipeline(data):

    audio_array, sample_rate = data
    return torch.tensor(audio_array, dtype=torch.float32), sample_rate

def text_pipeline(labels):

    return labels

# Add a dynamic pipeline to the dataset and specify the data fields provided by each function
train_data.add_dynamic_item(audio_pipeline, takes=["audio_samples"], provides=["processed_audio", "sample_rate"])
train_data.add_dynamic_item(text_pipeline, takes=["labels"], provides=["processed_text"])
valid_data.add_dynamic_item(audio_pipeline, takes=["audio_samples"], provides=["processed_audio", "sample_rate"])
valid_data.add_dynamic_item(text_pipeline, takes=["labels"], provides=["processed_text"])

# Set the output field to match the supplied key
train_data.set_output_keys(["processed_audio", "sample_rate", "processed_text"])
valid_data.set_output_keys(["processed_audio", "sample_rate", "processed_text"])

# Defining the loader
train_loader = SaveableDataLoader(train_data, batch_size=6, shuffle=True,collate_fn=collate_fn)
valid_loader = SaveableDataLoader(valid_data, batch_size=6, shuffle=False,collate_fn=collate_fn)


asr_model = ASR()

# Defining the Optimizer
asr_model.optimizer = torch.optim.Adam(asr_model.modules['wav2vec2'].parameters(), lr=1e-4)

asr_model.fit(
    epoch_counter=range(3),
    train_set=train_loader,
    valid_set=valid_loader
)



