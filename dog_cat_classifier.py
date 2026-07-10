"""
Dog vs. Cat Image Classifier

Author: Paola Hintze
Description:
This program builds a simple neural network that classifies
CIFAR-10 images as either cats or dogs.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers
from keras.datasets import cifar10


# Set random seeds to make the results more reproducible
np.random.seed(42)
tf.random.set_seed(42)


print("=" * 50)
print("ENVIRONMENT CHECK")
print("=" * 50)

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")
print(f"Available GPUs: {tf.config.list_physical_devices('GPU')}")

print("\nEnvironment setup completed successfully.")

# Load the CIFAR-10 dataset
print("\n" + "=" * 50)
print("LOADING AND PREPARING THE DATA")
print("=" * 50)

print("Loading CIFAR-10 dataset...")

(x_train_full, y_train_full), (x_test_full, y_test_full) = cifar10.load_data()

print(
    f"Full dataset shape - Training: {x_train_full.shape}, "
    f"Test: {x_test_full.shape}"
)

print(
    f"Full dataset labels - Training: {y_train_full.shape}, "
    f"Test: {y_test_full.shape}"
)

# CIFAR-10 class names
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

print("\nCIFAR-10 classes:")

for index, class_name in enumerate(class_names):
    print(f"  {index}: {class_name}")

# Select only cats and dogs
cat_class = 3
dog_class = 5

# Filter the training set
train_cat_mask = y_train_full.flatten() == cat_class
train_dog_mask = y_train_full.flatten() == dog_class
train_mask = train_cat_mask | train_dog_mask

x_train = x_train_full[train_mask]
y_train = y_train_full[train_mask]

# Convert labels to binary values:
# Cat = 0
# Dog = 1
y_train = np.where(y_train == cat_class, 0, 1).flatten()

# Filter the test set
test_cat_mask = y_test_full.flatten() == cat_class
test_dog_mask = y_test_full.flatten() == dog_class
test_mask = test_cat_mask | test_dog_mask

x_test = x_test_full[test_mask]
y_test = y_test_full[test_mask]

# Convert labels to binary values:
# Cat = 0
# Dog = 1
y_test = np.where(y_test == cat_class, 0, 1).flatten()

print("\nFiltered dataset:")
print(f"  Training images: {x_train.shape[0]}")
print(f"  Test images: {x_test.shape[0]}")
print(f"  Image shape: {x_train.shape[1:]}")

print("\nClass distribution:")
print(f"  Training cats: {(y_train == 0).sum()}")
print(f"  Training dogs: {(y_train == 1).sum()}")
print(f"  Test cats: {(y_test == 0).sum()}")
print(f"  Test dogs: {(y_test == 1).sum()}")

# Normalize pixel values from 0-255 to 0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("\nPixel value range after normalization:")
print(f"  Minimum value: {x_train.min():.1f}")
print(f"  Maximum value: {x_train.max():.1f}")

# Flatten each image from 32 x 32 x 3 into 3,072 features
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

print("\nFlattened data shapes:")
print(f"  Training data: {x_train_flat.shape}")
print(f"  Test data: {x_test_flat.shape}")

# Display ten sample images
figure, axes = plt.subplots(2, 5, figsize=(12, 6))

for index in range(10):
    row = index // 5
    column = index % 5

    axes[row, column].imshow(x_train[index])

    label = "Cat" if y_train[index] == 0 else "Dog"

    axes[row, column].set_title(label)
    axes[row, column].axis("off")

plt.tight_layout()
plt.savefig("sample_images.png", dpi=150, bbox_inches="tight")

print("\nSample images saved as 'sample_images.png'.")

plt.show()

print("\nData preparation completed successfully.")


# Build the neural network
print("\n" + "=" * 50)
print("BUILDING THE NEURAL NETWORK")
print("=" * 50)

model = keras.Sequential([
    keras.Input(shape=(3072,), name="input_layer"),

    layers.Dense(
        128,
        activation="relu",
        name="hidden_layer_1"
    ),

    layers.Dense(
        64,
        activation="relu",
        name="hidden_layer_2"
    ),

    layers.Dense(
        1,
        activation="sigmoid",
        name="output_layer"
    )
])

# Compile the model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nModel architecture:")
model.summary()

total_parameters = model.count_params()

print(f"\nTotal parameters: {total_parameters:,}")
print("\nNeural network built successfully.")

# Train the neural network
print("\n" + "=" * 50)
print("TRAINING THE MODEL")
print("=" * 50)

# Training hyperparameters
EPOCHS = 20
BATCH_SIZE = 32

print(f"Training epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Training samples: {len(x_train_flat)}")
print(f"Validation samples: {len(x_test_flat)}")

history = model.fit(
    x_train_flat,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_test_flat, y_test),
    verbose=1
)

print("\nTraining completed successfully.")

# Evaluate the trained model
print("\n" + "=" * 50)
print("EVALUATING THE MODEL")
print("=" * 50)

test_loss, test_accuracy = model.evaluate(
    x_test_flat,
    y_test,
    verbose=0
)

print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test accuracy percentage: {test_accuracy * 100:.2f}%")

# Generate predictions for the test set
predictions = model.predict(
    x_test_flat,
    verbose=0
)

predicted_classes = (predictions > 0.5).astype(int).flatten()

# Manually calculate accuracy
correct_predictions = (
    predicted_classes == y_test
).sum()

total_predictions = len(y_test)

manual_accuracy = (
    correct_predictions / total_predictions
)

print("\nManual accuracy check:")
print(
    f"Correct predictions: "
    f"{correct_predictions}/{total_predictions}"
)

print(
    f"Manual accuracy: "
    f"{manual_accuracy * 100:.2f}%"
)

# Plot training history
figure, axes = plt.subplots(
    1,
    2,
    figsize=(12, 4)
)

# Accuracy plot
axes[0].plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

axes[0].plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].set_title("Model Accuracy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss plot
axes[1].plot(
    history.history["loss"],
    label="Training Loss"
)

axes[1].plot(
    history.history["val_loss"],
    label="Validation Loss"
)

axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].set_title("Model Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "training_history.png",
    dpi=150,
    bbox_inches="tight"
)

print(
    "\nTraining history saved as "
    "'training_history.png'."
)

plt.show()

# Display sample predictions
print("\n" + "=" * 50)
print("SAMPLE PREDICTIONS")
print("=" * 50)

figure, axes = plt.subplots(
    2,
    5,
    figsize=(12, 6)
)

for index in range(10):
    row = index // 5
    column = index % 5

    prediction_value = predictions[index][0]

    predicted_label = (
        "Dog"
        if prediction_value > 0.5
        else "Cat"
    )

    actual_label = (
        "Dog"
        if y_test[index] == 1
        else "Cat"
    )

    confidence = (
        prediction_value
        if prediction_value > 0.5
        else 1 - prediction_value
    )

    is_correct = (
        predicted_classes[index] == y_test[index]
    )

    result_symbol = (
        "Correct"
        if is_correct
        else "Incorrect"
    )

    axes[row, column].imshow(
        x_test[index]
    )

    axes[row, column].set_title(
        f"Predicted: {predicted_label}\n"
        f"Actual: {actual_label}\n"
        f"Confidence: {confidence:.2f}\n"
        f"{result_symbol}"
    )

    axes[row, column].axis("off")

plt.tight_layout()

plt.savefig(
    "sample_predictions.png",
    dpi=150,
    bbox_inches="tight"
)

print(
    "Sample predictions saved as "
    "'sample_predictions.png'."
)

plt.show()

print("\nModel evaluation completed successfully.")

# Export the trained model
print("\n" + "=" * 50)
print("EXPORTING THE MODEL")
print("=" * 50)

import os

# Create a folder for the exported model files
os.makedirs("saved_models", exist_ok=True)

# Save the complete model in the recommended Keras format
full_model_path = "saved_models/dog_cat_classifier.keras"

model.save(full_model_path)

full_model_size = (
    os.path.getsize(full_model_path)
    / (1024 * 1024)
)

print("\nFull model saved successfully.")
print(f"File path: {full_model_path}")
print(f"File size: {full_model_size:.2f} MB")

# Save only the trained weights
weights_path = (
    "saved_models/"
    "dog_cat_classifier.weights.h5"
)

model.save_weights(weights_path)

weights_size = (
    os.path.getsize(weights_path)
    / (1024 * 1024)
)

print("\nModel weights saved successfully.")
print(f"File path: {weights_path}")
print(f"File size: {weights_size:.2f} MB")

# Export the model in TensorFlow SavedModel format
savedmodel_path = (
    "saved_models/"
    "dog_cat_classifier_savedmodel"
)

model.export(savedmodel_path)

print("\nTensorFlow SavedModel exported successfully.")
print(f"Directory path: {savedmodel_path}")

# Load the complete model again
print("\nVerifying the saved model...")

loaded_model = keras.models.load_model(
    full_model_path
)

loaded_test_loss, loaded_test_accuracy = (
    loaded_model.evaluate(
        x_test_flat,
        y_test,
        verbose=0
    )
)

print(
    f"Original model accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Loaded model accuracy: "
    f"{loaded_test_accuracy * 100:.2f}%"
)

accuracy_matches = (
    abs(
        loaded_test_accuracy
        - test_accuracy
    )
    < 0.001
)

print(
    "Accuracy match: "
    + (
        "Yes"
        if accuracy_matches
        else "No"
    )
)

print("\nModel export completed successfully.")

# Experiment with different hyperparameters
print("\n" + "=" * 50)
print("HYPERPARAMETER EXPERIMENTS")
print("=" * 50)


def build_experiment_model(hidden_layers, learning_rate):
    experiment_model = keras.Sequential()

    experiment_model.add(
        keras.Input(shape=(3072,))
    )

    for neurons in hidden_layers:
        experiment_model.add(
            layers.Dense(
                neurons,
                activation="relu"
            )
        )

    experiment_model.add(
        layers.Dense(
            1,
            activation="sigmoid"
        )
    )

    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    experiment_model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return experiment_model


experiments = [
    {
        "name": "Experiment 1",
        "hidden_layers": [64],
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10
    },
    {
        "name": "Experiment 2",
        "hidden_layers": [128, 64],
        "learning_rate": 0.0001,
        "batch_size": 32,
        "epochs": 20
    },
    {
        "name": "Experiment 3",
        "hidden_layers": [256, 128, 64],
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 20
    }
]

experiment_results = []

for experiment in experiments:
    print("\n" + "-" * 50)
    print(experiment["name"])
    print("-" * 50)

    print(
        f"Hidden layers: "
        f"{experiment['hidden_layers']}"
    )

    print(
        f"Learning rate: "
        f"{experiment['learning_rate']}"
    )

    print(
        f"Batch size: "
        f"{experiment['batch_size']}"
    )

    print(
        f"Epochs: "
        f"{experiment['epochs']}"
    )

    experiment_model = build_experiment_model(
        hidden_layers=experiment["hidden_layers"],
        learning_rate=experiment["learning_rate"]
    )

    experiment_history = experiment_model.fit(
        x_train_flat,
        y_train,
        validation_data=(
            x_test_flat,
            y_test
        ),
        batch_size=experiment["batch_size"],
        epochs=experiment["epochs"],
        verbose=1
    )

    experiment_loss, experiment_accuracy = (
        experiment_model.evaluate(
            x_test_flat,
            y_test,
            verbose=0
        )
    )

    experiment_results.append({
        "name": experiment["name"],
        "hidden_layers": experiment["hidden_layers"],
        "learning_rate": experiment["learning_rate"],
        "batch_size": experiment["batch_size"],
        "epochs": experiment["epochs"],
        "test_loss": experiment_loss,
        "test_accuracy": experiment_accuracy
    })

    print(
        f"\n{experiment['name']} "
        f"test accuracy: "
        f"{experiment_accuracy * 100:.2f}%"
    )


print("\n" + "=" * 50)
print("EXPERIMENT RESULTS")
print("=" * 50)

for result in experiment_results:
    print(f"\n{result['name']}")
    print(
        f"  Hidden layers: "
        f"{result['hidden_layers']}"
    )
    print(
        f"  Learning rate: "
        f"{result['learning_rate']}"
    )
    print(
        f"  Batch size: "
        f"{result['batch_size']}"
    )
    print(
        f"  Epochs: "
        f"{result['epochs']}"
    )
    print(
        f"  Test loss: "
        f"{result['test_loss']:.4f}"
    )
    print(
        f"  Test accuracy: "
        f"{result['test_accuracy'] * 100:.2f}%"
    )

best_experiment = max(
    experiment_results,
    key=lambda result: result["test_accuracy"]
)

print("\nBest experiment:")
print(best_experiment["name"])
print(
    f"Best test accuracy: "
    f"{best_experiment['test_accuracy'] * 100:.2f}%"
)

print("\nHyperparameter experiments completed successfully.")