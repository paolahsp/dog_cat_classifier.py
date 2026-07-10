# Learning Reflection

## Teaching Models to See: Images

### How did the neural network learn to distinguish dogs from cats?

The neural network learned by comparing thousands of labeled cat and dog images. During training, it made a prediction for each image, compared that prediction with the correct label, and calculated an error. It then adjusted its internal weights to reduce that error. This process was repeated over many epochs, allowing the model to gradually improve its predictions.

### What patterns do you think the model learned?

The model probably learned combinations of colors, edges, textures, shapes, and spatial patterns that were common in cat and dog images. For example, it may have learned differences related to ears, faces, fur patterns, body shapes, and backgrounds. However, because the images were flattened before being sent to the model, it did not preserve spatial relationships as effectively as a convolutional neural network would.

### Why did flattening the image work, and what information might be lost?

Flattening worked because every pixel value was still included as an input feature. The model received all 3,072 numerical values from each image and could learn relationships between them.

However, flattening removed the original two-dimensional structure of the image. The model no longer directly understood which pixels were next to each other or which groups of pixels formed shapes. This made it harder to recognize visual patterns and limited the model's accuracy.

## Teaching Models to Understand: Audio

### How is audio different from images as input to a neural network?

Images are organized across width, height, and color channels, while audio is organized as a sequence of signal values over time. In an image, spatial position is important. In audio, timing, order, rhythm, pronunciation, and changes in frequency are important.

This means that audio models must understand how sounds develop over time, while image models focus more on spatial patterns.

### What do you think the pre-trained audio model learned during training?

The pre-trained model likely learned speech patterns, pronunciation, sound features, timing, and relationships between spoken words and user intentions. It did not only learn individual sounds. It also learned how combinations of sounds and words represented banking-related requests such as paying a bill, freezing an account, checking a balance, or reporting an app error.

### Why do we use pre-trained models instead of training from scratch?

Pre-trained models have already learned useful features from large datasets. Training an audio transformer from scratch would require a large amount of labeled audio, powerful hardware, and a long training time.

Using a pre-trained model is faster, less expensive, and often produces better results, especially when the available dataset is small.

## Transfer Learning

### The audio model was trained on one dataset but works on another. Why?

The model learned general speech representations during pre-training. These representations include useful information about sounds, pronunciation, timing, and language patterns. Because those features are also present in new audio samples, the model can apply its previous knowledge to related tasks.

### What knowledge transferred from the original training?

The transferred knowledge included how to process raw audio, identify important sound patterns, recognize speech features, and connect spoken language with meaning. The final classification layers then used those learned features to identify specific user intents.

### How is this similar to how humans learn?

Humans also reuse previous knowledge. For example, after learning one language or recognizing certain speech patterns, a person can use that experience to learn related languages or understand new accents more easily. We do not begin every new task without prior knowledge.

## Model Architecture

### Why are the image and audio models structured differently?

The two models process different types of information. The image model received fixed-size pixel values, while the audio model processed a time-based sequence.

Different data types have different structures, so the architecture must match the type of patterns the model needs to learn.

### What makes transformers useful for audio and text sequences?

Transformers are useful because they can study relationships between different parts of a sequence. Their attention mechanism helps the model identify which sections of the input are most important and how distant parts relate to each other.

For audio, this helps the model connect sounds across time. For text, it helps connect words across a sentence.

### How do convolutional layers differ from transformer layers?

Convolutional layers focus on local patterns. In images, they detect nearby features such as edges, shapes, and textures. They process small regions and build more complex features across deeper layers.

Transformer layers use attention to compare different parts of the input, including parts that may be far apart. This makes them especially useful for sequences where long-range context matters.

## Neural Network Basics

### What are the key components of a neural network?

The main components are input data, layers, neurons, weights, biases, activation functions, a loss function, and an optimizer.

The layers transform the data, the activation functions introduce non-linearity, the loss function measures the model's error, and the optimizer adjusts the weights to reduce that error.

## Training Process

### What happens during training?

During training, the model receives input data and makes predictions. It compares those predictions with the correct labels and calculates the loss.

Backpropagation determines how much each weight contributed to the error, and the optimizer updates the weights. Repeating this process across many examples and epochs allows the model to learn.

## Hyperparameters

### Which parameters had the biggest impact on performance?

The learning rate and model architecture had the biggest impact in this experiment.

Experiment 2 achieved the best test accuracy at 62.40%. It used two hidden layers with 128 and 64 neurons and a smaller learning rate of 0.0001.

The deeper model in Experiment 3 did not perform better, which showed that adding more layers and neurons does not automatically improve generalization.

## Pre-trained Models

### What are the advantages of pre-trained models?

Pre-trained models save time, reduce computing requirements, and often perform well with limited data. They also allow developers to use knowledge learned from large datasets without recreating the entire training process.

### When would you train a model from scratch?

Training from scratch may be useful when the data is very different from existing models, when a highly specialized architecture is required, or when there is a very large custom dataset available.

It may also be necessary when privacy, licensing, or domain-specific requirements prevent the use of an existing model.

## Different Data Types

### How do models process images and audio differently?

Image models process spatial information, including position, shape, texture, and color. Audio models process values across time and focus on sound frequency, rhythm, pronunciation, and sequence order.

Both types of models convert raw data into numerical features and learn patterns by adjusting weights.

## Model Understanding

### How do models “see” images and “understand” audio?

Models do not see or understand in the same way humans do. They process numerical patterns.

For images, the numbers represent pixel colors and positions. For audio, the numbers represent changes in the sound waveform over time.

The model learns statistical relationships between these numbers and the labels provided during training.
