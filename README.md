# Comparison of CNN, and RNN for text classification using Pytorch


In this repo I will explore non-linear text classification algorithms : Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs). 
Two datasets have been used for classification task:

•	The first dataset is a subset of a Clickbait Dataset that has article headlines and a binary label on whether the headline is considered clickbait.
•	The second dataset is a subset of Web of Science Dataset that has articles and a corresponding label on the domain of the articles.


I will first explore Continuous Bag-of-Words (CBOW) and Skip-gram based Word2Vec models using a very small dataset. I will then use pre-trained Word2Vec embeddings and feed them to the classification algorithms.

## Word2Vec 

Word2vec is a method to efficiently create word embeddings. More details on word2vec and the intuition behind it can be found here :  
* [The Illustrated Word2vec by Jay Alammar](https://jalammar.github.io/illustrated-word2vec/)

Word2vec is based on the idea that a word’s meaning is defined by its context. Context is represented as surrounding words. For the word2vec model, context is represented as N words before and N words after the current word. N is a hyperparameter. With larger N we can create better embeddings, but at the same time, such a model requires more computational resources. 

There are two word2vec architectures proposed in the paper:

* CBOW (Continuous Bag-of-Words) — a model that predicts a current word based on its context words.
* Skip-Gram — a model that predicts context words based on the current word.

## Text Classification with CNN
Convolutional layers are used to find patterns by sliding small kernel window over input. Instead of multiplying the filters on the small regions of the images, it slides through embedding vectors of few words as mentioned by window size. For looking at sequences of word embeddings, the window has to look at multiple word embeddings in a sequence. They will be rectangular with size window_size * embedding_size. For example, if window size is 3 then kernel will be 3*500. This essentially represents n-grams in the model. The kernel weights (filter) are multiplied to word embeddings in pairs and summed up to get output values. As the network is being learned, these kernel weights are also being learned.

I will be using convolutional network with pre-trained word2vec models for classification.Pre-trained word2vec models have been used for feasibility of finding appropriate embeddings. 

## Text Classification with RNN

Recurrent NAural Networks are suitable for sequence data(text or time series) where the output of one step would be fed as input to the next step.

I will be using recurrent neural networks with pre-trained word2vec models for classification. Pre-trained word2vec models have been used for feasibility of finding appropriate embeddings.
