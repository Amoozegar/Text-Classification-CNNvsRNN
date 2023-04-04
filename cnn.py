import torch
torch.manual_seed(10)
import torch.nn.functional as F
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, w2vmodel, num_classes, window_sizes=(1,2,3,5)):
        '''
        Initialize CNN with the embedding layer, convolutional layer and a linear layer.

        Args:
            w2vmodel: Pre-trained word2vec model.
            num_classes: Number of classes (labels).
            window_sizes: Window size for the convolution kernel.
        '''
        super(CNN, self).__init__()
        weights = w2vmodel.wv # use this to initialize the embedding layer
        EMBEDDING_SIZE = 500  # Use this to set the embedding_dim in embedding layer
        NUM_FILTERS = 10      # Number of filters in CNN
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights.vectors), padding_idx=w2vmodel.wv.vocab['pad'].index)
        self.embedding.weight.requires_grad = False

        self.conv_layers = nn.ModuleList(nn.Conv2d(1, out_channels = NUM_FILTERS, kernel_size = (window, w2vmodel.wv.vector_size),padding = (window-1, 0)) for window in window_sizes )
        self.linear_layer = nn.Linear( len(window_sizes) * NUM_FILTERS, num_classes)
    def forward(self, x):
        '''
        Implement the forward function to feed the input through the model and get the output.
        Args:
            inputs: Input data.

        Returns:
            output: Probabilities of each label.
        '''

        embedded = self.embedding(x)
        embedded = embedded.unsqueeze(1)
        conv_out=[]
        for conv in self.conv_layers:
            convoluted = nn.functional.tanh(conv(embedded)).squeeze(3)  #output : (N,Cout, Hout, Wout)
            max_pool_out = nn.functional.max_pool1d(convoluted, convoluted.size(2)).squeeze(2)  #input : (minibatch,in_channels,iW)
            conv_out.append(max_pool_out)
        out = torch.cat(conv_out, 1)
        output = self.linear_layer(out)
        return nn.functional.softmax(output)
