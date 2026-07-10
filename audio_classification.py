"""
Audio Intent Classification with Transformers

Author: Paola Hintze
Description:
This program uses a pre-trained transformer model to classify
spoken audio samples by their intended meaning.
"""

import torch
import torchaudio
import datasets
import transformers

from datasets import Audio
from datasets import load_dataset
from transformers import pipeline


print("=" * 50)
print("AUDIO CLASSIFICATION ENVIRONMENT CHECK")
print("=" * 50)

print(f"PyTorch version: {torch.__version__}")
print(f"Torchaudio version: {torchaudio.__version__}")
print(f"Datasets version: {datasets.__version__}")
print(f"Transformers version: {transformers.__version__}")

print(f"CUDA available: {torch.cuda.is_available()}")

device_name = (
    torch.cuda.get_device_name(0)
    if torch.cuda.is_available()
    else "CPU"
)

print(f"Selected device: {device_name}")

print("\nAudio classification environment is ready.")

# Load the audio dataset
print("\n" + "=" * 50)
print("LOADING THE AUDIO DATASET")
print("=" * 50)

print("Downloading and loading the PolyAI MINDS-14 dataset...")

minds = load_dataset(
    "PolyAI/minds14",
    name="en-AU",
    split="train"
)

# Convert the audio samples to a 16,000 Hz sampling rate
minds = minds.cast_column(
    "audio",
    Audio(sampling_rate=16_000)
)

print("\nDataset loaded successfully.")

print(f"Number of audio samples: {len(minds)}")
print(f"Dataset columns: {minds.column_names}")

print("\nDataset structure:")
print(minds.features)

# Examine the first sample
example = minds[0]

print("\nFirst sample:")
print(f"Audio path: {example['audio']['path']}")
print(f"Sampling rate: {example['audio']['sampling_rate']}")
print(f"Number of audio values: {len(example['audio']['array'])}")
print(f"Intent class number: {example['intent_class']}")

# Convert the numeric class into a readable label
id2label = minds.features["intent_class"].int2str

actual_label = id2label(
    example["intent_class"]
)

print(f"Actual intent label: {actual_label}")

print("\nAudio dataset preparation completed successfully.")

# Load a pre-trained audio classification model
print("\n" + "=" * 50)
print("LOADING THE PRE-TRAINED AUDIO CLASSIFIER")
print("=" * 50)

model_name = "anton-l/xtreme_s_xlsr_300m_minds14"

print(f"Model: {model_name}")
print("Loading the model from Hugging Face...")

audio_classifier = pipeline(
    task="audio-classification",
    model=model_name,
    device=-1
)

print("\nPre-trained model loaded successfully.")

# Classify the first audio sample
audio_input = {
    "array": example["audio"]["array"],
    "sampling_rate": example["audio"]["sampling_rate"]
}

predictions = audio_classifier(
    audio_input,
    top_k=5
)

print("\nPrediction results:")

for rank, prediction in enumerate(predictions, start=1):
    print(
        f"{rank}. "
        f"{prediction['label']} - "
        f"{prediction['score'] * 100:.2f}%"
    )

predicted_label = predictions[0]["label"]
predicted_confidence = predictions[0]["score"]

print("\nComparison:")
print(f"Actual label: {actual_label}")
print(f"Predicted label: {predicted_label}")
print(
    f"Prediction confidence: "
    f"{predicted_confidence * 100:.2f}%"
)

prediction_correct = (
    predicted_label == actual_label
)

print(
    "Prediction result: "
    + (
        "Correct"
        if prediction_correct
        else "Incorrect"
    )
)

print("\nSingle audio classification completed successfully.")

# Test the classifier on multiple audio samples
print("\n" + "=" * 50)
print("TESTING MULTIPLE AUDIO SAMPLES")
print("=" * 50)

NUMBER_OF_SAMPLES = 10
RANDOM_SEED = 42

# Shuffle the dataset so the samples are not taken
# from only one intent category
shuffled_minds = minds.shuffle(seed=RANDOM_SEED)

correct_predictions = 0
sample_results = []

for index in range(NUMBER_OF_SAMPLES):
    sample = shuffled_minds[index]

    sample_audio = {
        "array": sample["audio"]["array"],
        "sampling_rate": sample["audio"]["sampling_rate"]
    }

    sample_actual_label = id2label(
        sample["intent_class"]
    )

    sample_predictions = audio_classifier(
        sample_audio,
        top_k=1
    )

    sample_predicted_label = (
        sample_predictions[0]["label"]
    )

    sample_confidence = (
        sample_predictions[0]["score"]
    )

    is_correct = (
        sample_predicted_label
        == sample_actual_label
    )

    if is_correct:
        correct_predictions += 1

    sample_results.append({
        "sample": index + 1,
        "actual": sample_actual_label,
        "predicted": sample_predicted_label,
        "confidence": sample_confidence,
        "correct": is_correct
    })

    print(f"\nSample {index + 1}")
    print(f"Actual label: {sample_actual_label}")
    print(f"Predicted label: {sample_predicted_label}")
    print(f"Confidence: {sample_confidence * 100:.2f}%")

    print(
        "Result: "
        + (
            "Correct"
            if is_correct
            else "Incorrect"
        )
    )

overall_accuracy = (
    correct_predictions
    / NUMBER_OF_SAMPLES
)

unique_labels = {
    result["actual"]
    for result in sample_results
}

print("\n" + "=" * 50)
print("MULTIPLE SAMPLE RESULTS")
print("=" * 50)

print(
    f"Correct predictions: "
    f"{correct_predictions}/{NUMBER_OF_SAMPLES}"
)

print(
    f"Overall accuracy: "
    f"{overall_accuracy * 100:.2f}%"
)

print(
    f"Different actual labels tested: "
    f"{len(unique_labels)}"
)

print("\nLabels included in the test:")

for label in sorted(unique_labels):
    print(f"  - {label}")

print(
    "\nMultiple audio sample testing "
    "completed successfully."
)