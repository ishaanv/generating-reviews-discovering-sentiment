# Unsupervised Sentiment Polarity Classification of Text

Classify quantitaive sentiment polarity using the trained network from [Learning to Generate Reviews and Discovering Sentiment](https://github.com/openai/generating-reviews-discovering-sentiment) (Alec Radford, Rafal Jozefowicz, Ilya Sutskever).

### Architecture and Data:
This repo also contains the parameters of the multiplicative LSTM model with 4,096 units we trained on the Amazon product review dataset introduced in McAuley et al. (2015) [1]. The dataset in de-duplicated form contains over 82 million product reviews from May 1996 to July 2014 amounting to over 38 billion training bytes. Training took one month across four NVIDIA Pascal GPUs, with our model processing 12,500 characters per second.

[1] McAuley, Julian, Pandey, Rahul, and Leskovec, Jure. Inferring networks of substitutable and complementary products. In *Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 785–794. ACM, 2015.

### Classifying Text

In email_sentiment_classifier.py,
```
def calculate_sentiment(body, messageid=None, debug=False):
    # sentiment neuron
    sentiment = model.transform([body])[0, sentiment_neuron]
    if debug and messageid:
        print("id: {} -> calculated senitment: {}".format(messageid, sentiment))
    return sentiment
 ```
 simply returns the polarity score given some text.

 ### Performance and Assumptions
 The LSTM was trained on Amazon review. This should be kept in mind as your score are a function of this training data especially when used unsupervised.
 