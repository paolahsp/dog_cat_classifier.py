## Multiple Audio Sample Evaluation

I tested the pre-trained audio classification model on 10 shuffled samples from the MINDS-14 dataset.

The shuffled samples represented several different intent categories, including address, app error, cash deposit, card issues, balance, high-value payment, business loan, freeze, and latest transactions.

The model correctly classified all 10 samples.

- Correct predictions: 10 out of 10
- Overall accuracy: 100.00%
- Confidence scores: approximately 99.66% to 99.91%

These results show that the pre-trained transformer model was highly effective at recognizing different banking-related audio intents. The high confidence scores also indicate that the model was very certain about its predictions.

However, this result is based on a small sample of only 10 audio recordings, so it should not be interpreted as proof that the model will always achieve 100% accuracy on the full dataset or on completely new audio.