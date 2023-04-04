import torch
torch.manual_seed(10)
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RNN(nn.Module):
    def __init__(self, vocab, num_classes):
        '''
        Initialize RNN with the embedding layer, bidirectional RNN layer and a linear layer with a dropout.
    
        Args:
        vocab: Vocabulary.
        num_classes: Number of classes (labels).
        # embed_len : embedding_dim default value for embedding layer
        # hidden_dim : hidden_dim default value for rnn layer
        # n_layers : number of layers for RNN
        #
        '''
        super(RNN, self).__init__()
        self.embed_len = 50  # embedding_dim default value for embedding layer
        self.hidden_dim = 75 # hidden_dim default value for rnn layer
        self.n_layers = 1    # num_layers default value for rnn

        dropout_rate =0.5
        self.embedding = nn.Embedding(len(vocab), self.embed_len)
        self.rnn = nn.RNN(input_size=self.embed_len, hidden_size=self.hidden_dim, num_layers=self.n_layers, batch_first=True, bidirectional=True) #bach size first
        self.linear = nn.Linear(2*self.hidden_dim, num_classes)
        self.dropout = nn.Dropout(p=dropout_rate)


    def forward(self, inputs, inputs_len):
        '''
        Implement the forward function to feed the input through the model and get the output.

        Args:
        inputs: Input data.
        inputs_len: Length of inputs in each batch.

        Returns:
            output: Logits of each label.
        '''

        embedded = self.embedding(inputs)  # apply dropout to the embeddings
        packed_embedded = pack_padded_sequence(embedded, inputs_len, batch_first=True,enforce_sorted=False) #batch size first
        packed_output,h_n = self.rnn(packed_embedded) # h_n : the final hidden state for each element in the batch
        seq_unpacked, lens_unpacked = pad_packed_sequence(packed_output, batch_first=True) #lens_unpacked: list of lengths of each sequence in the batch
        last_forward_hidden_state = seq_unpacked[:, -1, :self.rnn.hidden_size]
        first_backward_hidden_state = seq_unpacked[:, 0, self.rnn.hidden_size:]
        concat_hidden = torch.cat([last_forward_hidden_state, first_backward_hidden_state], dim=1) #  concatenate the hidden states from two sides (LTR, and RTL)
        linear_output = self.linear(concat_hidden)
        return linear_output


