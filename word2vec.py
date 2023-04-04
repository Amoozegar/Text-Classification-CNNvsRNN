import numpy as np
import torch

torch.manual_seed(10)
import torch.nn.functional as F
import torch.nn as nn


class Word2Vec(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.vocabulary_size = 0

    def tokenize(self, data):
        """
        Split all the words in the data into tokens.

        Args:
            data: (N,) list of sentences in the dataset.

        Return:
            tokens: (N, D_i) list of tokenized sentences. The i-th sentence has D_i words after the split.
        """
        tokens = []
        for sentence in data:
            sentence = sentence.strip()
            words = sentence.split()
            tokens.append(words)

        return tokens

    def create_vocabulary(self, tokenized_data):
        """
        Create a vocabulary for the tokenized data.

        Args:
            tokenized_data: (N, D) list of split tokens in each sentence.
            
        Return:
            None (The update is done for self.word2idx, self.idx2word and self.vocabulary_size)
        """
        vocab = set()
        for sentence in tokenized_data:
            for word in sentence:
                if word not in vocab:
                    vocab.add(word)
        vocab_list = list(vocab)
        vocab_list.sort()

        self.word2idx = dict(zip(vocab_list, range(len(vocab_list))))
        self.idx2word = dict(zip(range(len(vocab_list)), vocab_list))
        self.vocabulary_size = len(vocab_list)

    def skipgram_embeddings(self, tokenized_data, window_size=2):
        """
        Create a skipgram embeddings by taking context as middle word and predicting
        N=window_size past words and N=window_size future words.

        [1, 2, 3, 4, 5] ->
           source_tokens             target_tokens
           [1]                       [2]
           [1]                       [3]
           [2]                       [1]
           [2]                       [3]
           [2]                       [4]
           [3]                       [1]
           [3]                       [2]
           [3]                       [4]
           [3]                       [5]
           [4]                       [2]
           [4]                       [3]
           [4]                       [5]
           [5]                       [3]
           [5]                       [4]

        source_tokens: [[1], [1], [2], [2], [2], ...]
        target_tokens: [[2], [3], [1], [3], [4], ...]
        Args:
            tokenized_data: (N, D_i) list of split tokens in each sentence.
            window_size: length of the window for creating context. Default is 2.

        Returns:
            source_tokens: List of elements where each element is the middle word in the window.
            target_tokens: List of elements representing IDs of the context words.
        """
        source_tokens = []
        target_tokens = []
        for sentence in tokenized_data:
            for i in range(len(sentence)):
                source = [self.word2idx.get(sentence[i])]
                target_list = [self.word2idx.get(sentence[j]) for j in range(i - window_size, i + window_size + 1) if (i != j) and (j >= 0) and (j <= len(sentence)-1)]
                for target in target_list:
                    target_tokens.append([target])
                    source_tokens.append(source)
        return source_tokens, target_tokens


    def cbow_embeddings(self, tokenized_data, window_size=2):
        """
        Create a cbow embeddings by taking context as N=window_size past words and N=window_size future words.

        [1, 2, 3, 4, 5] ->
           source_tokens             target_tokens
           [2,3]                     [1]
           [1,3,4]                   [2]
           [1,2,4,5]                 [3]
           [2,3,5]                   [4]
           [3,4]                     [5]
           
        source_tokens: [[2,3], [1,3,4], [1,2,4,5], [2,3,5], [3,4]]
        target_tokens: [[1], [2], [3], [4], [5]]

        Args:
            tokenized_data: (N, D_i) list of split tokens in each sentence.
            window_size: length of the window for creating context. Default is 2.

        Returns:
            source_tokens: List of elements where each element is maximum of N=window_size*2 context word IDs.
            target_tokens: List of elements representing IDs of the middle word in the window.
        """
        source_tokens = []
        target_tokens = []
        for sentence in tokenized_data:
            for i in range(len(sentence)):
                target = [self.word2idx.get(sentence[i])]
                source = [self.word2idx.get(sentence[j]) for j in range(i - window_size, i + window_size + 1) if (i != j) and (j >= 0) and (j <= len(sentence)-1)]
                target_tokens.append(target)
                source_tokens.append(source)


        return source_tokens, target_tokens



class SkipGram_Model(nn.Module):
    def __init__(self, vocab_size: int):
        """
        Initialize SkipGram_Model with the embedding layer and a linear layer.

        Reference: 
            embedding - https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
            linear layer - https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

        Args:
            vocab_size: Size of the vocabulary.
        """
        super(SkipGram_Model, self).__init__()
        self.EMBED_DIMENSION = 300 # please use this to set embedding_dim in embedding layer
        self.EMBED_MAX_NORM = 1    # please use this to set max_norm in embedding layer
        self.embedding_layer = nn.Embedding(vocab_size, self.EMBED_DIMENSION, max_norm=self.EMBED_MAX_NORM)
        self.linear_layer = nn.Linear(self.EMBED_DIMENSION, vocab_size)


    def forward(self, inputs):
        """
        Implement the SkipGram model architecture as described in the notebook.

        Args:
            inputs: Tensor of IDs for each sentence.

        Returns:
            output: Tensor of logits with shape same as vocab_size.

        """
        embedded = self.embedding_layer(inputs)
        output = self.linear_layer(embedded)
        return output


class CBOW_Model(nn.Module):
    def __init__(self, vocab_size: int):
        """
        Initialize CBOW_Model with the embedding layer and a linear layer.

        Reference: 
            embedding - https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
            linear layer - https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

        Args:
            vocab_size: Size of the vocabulary.
        """
        super(CBOW_Model, self).__init__()
        self.EMBED_DIMENSION = 300  # please use this to set embedding_dim in embedding layer
        self.EMBED_MAX_NORM = 1     # please use this to set max_norm in embedding layer
        self.embedding_layer = torch.nn.Embedding(vocab_size, self.EMBED_DIMENSION, max_norm=self.EMBED_MAX_NORM)
        self.linear_layer = torch.nn.Linear(self.EMBED_DIMENSION, vocab_size)


    def forward(self, inputs):
        """
        Implement the CBOW model architecture as described in the notebook.

        Args:
            inputs: Tensor of IDs for each sentence.

        Returns:
            output: Tensor of logits with shape same as vocab_size.

        """

        averaged = torch.mean(self.embedding_layer(inputs), dim=0, keepdim=True)
        output = self.linear_layer(averaged)
        return output
