from speechbrain.dataio.dataloader import SaveableDataLoader
from jiwer import wer  

from prepare import load_tedlium_dataset
import torch
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.dataio.dataset import DynamicItemDataset
from train import extract_data_from_dataset
from train import prepare_data_dict
from train import audio_pipeline
from train import text_pipeline
from train import collate_fn
from train import ASR

tedlium = load_tedlium_dataset()
# Loading Test Sets
test_audio_samples, test_transcriptions = extract_data_from_dataset(tedlium["test"], subset_percent=10)

# Preparing the Test Data Dictionary
test_data_dict = prepare_data_dict(test_audio_samples, test_transcriptions)

# Creating a test dataset example
test_data = DynamicItemDataset(test_data_dict)

# Adding Dynamic Pipes
test_data.add_dynamic_item(audio_pipeline, takes=["audio_samples"], provides=["processed_audio", "sample_rate"])
test_data.add_dynamic_item(text_pipeline, takes=["labels"], provides=["processed_text"])

# Setting Output Fields
test_data.set_output_keys(["processed_audio", "sample_rate", "processed_text"])

# Creating a Test Set Loader
test_loader = SaveableDataLoader(test_data, batch_size=6, shuffle=False, collate_fn=collate_fn)

def evaluate_model(asr_model, test_loader):
    asr_model.modules.wav2vec2.eval()  
    all_predictions = []
    all_references = []

    with torch.no_grad():
        for batch in test_loader:

            audios,_, targets = batch

            input_values = asr_model.processor(
                audios.cpu().numpy(),
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            ).input_values.to(asr_model.device)

            logits = asr_model.modules.wav2vec2(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)


            predictions = asr_model.processor.batch_decode(predicted_ids, skip_special_tokens=True)
            all_predictions.extend(predictions)

            references = asr_model.processor.batch_decode(targets, skip_special_tokens=True)
            all_references.extend(references)

    # Calculate WER
    wer_score = wer(all_references, all_predictions)
    print("Word Error Rate (WER):", wer_score)
    return all_predictions, all_references

asr_model = ASR()


predictions, references = evaluate_model(asr_model, test_loader)


for i in range(5): 
    print(f"Reference: {references[i]}")
    print(f"Prediction: {predictions[i]}")
    print("-" * 50)