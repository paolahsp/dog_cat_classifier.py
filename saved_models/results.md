## Hyperparameter Experiments

I tested three different neural network configurations to understand how model depth, learning rate, batch size, and number of epochs affected performance.

### Experiment 1

- Hidden layers: [64]
- Learning rate: 0.001
- Batch size: 32
- Epochs: 10
- Test loss: 0.6681
- Test accuracy: 58.05%

This was the simplest model. It trained quickly, but its performance was the lowest. The single hidden layer and shorter training time may have limited the model's ability to learn more complex visual patterns.

### Experiment 2

- Hidden layers: [128, 64]
- Learning rate: 0.0001
- Batch size: 32
- Epochs: 20
- Test loss: 0.6601
- Test accuracy: 62.40%

This configuration achieved the best result. The two hidden layers provided enough capacity to learn useful patterns, while the smaller learning rate allowed the model to update its weights more gradually and consistently.

### Experiment 3

- Hidden layers: [256, 128, 64]
- Learning rate: 0.001
- Batch size: 64
- Epochs: 20
- Test loss: 0.6700
- Test accuracy: 60.90%

This deeper model performed better than Experiment 1 but slightly worse than Experiment 2. Adding more neurons and layers did not automatically improve performance. The larger model may have been more difficult to optimize and may also have started to overfit the training data.

### Conclusion

Experiment 2 produced the best test accuracy at 62.40%. This showed that model complexity alone does not guarantee better performance. A smaller learning rate and a balanced architecture helped the model generalize better to unseen images.