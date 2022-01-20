'''
Author: Zhengxiang (Jack) Wang 
GitHub: https://github.com/jaaack-wang 
About: Simple RNN model for text classification using pytorch. 
'''
import torch
import torch.nn as nn


class SimpleRNN(nn.Module):
    
    def __init__(self,
                 vocab_size,
                 output_dim, 
                 embedding_dim=100,
                 rnn_hidden_dim=128,
                 padding_idx=0, 
                 hidden_dim_out=50,
                 n_layers=1,
                 bidirectional=False,
                 dropout_rate=0.0,
                 activation=nn.ReLU()):
        
        super().__init__()
        
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx)
        
        self.bidirectional = bidirectional
        self.rnn = nn.RNN(embedding_dim, rnn_hidden_dim, n_layers, 
                          dropout=dropout_rate, bidirectional=self.bidirectional)
        
        rnn_out_dim = rnn_hidden_dim * 2 if self.bidirectional is True else rnn_hidden_dim
        self.dense = nn.Linear(rnn_out_dim, hidden_dim_out)
        self.activation = activation
        self.dense_out = nn.Linear(hidden_dim_out, output_dim)
    
    def encoder(self, embd, seq_len):
        embd = nn.utils.rnn.pack_padded_sequence(embd, 
                                                 lengths=seq_len, 
                                                 batch_first=True, 
                                                 enforce_sorted=False)
        # shape: encoded, (batch_size, seq_len, rnn_hidden_dim)
        # shape: hidden, (1, batch size, rnn_hidden_dim)
        encoded, hidden = self.rnn(embd)
        
        # shape: hidden, (batch size, rnn_out_dim);
        # if bidirectional, rnn_out_dim = rnn_hidden_dim * 2; otherwise same
        if self.bidirectional is False:  
#             return  hidden[-1]    # This works too
            return hidden[-1, :, :]
#         return torch.cat([hidden[-2], hidden[-1]], dim=-1)    # This works too
        return torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
    
    def forward(self, text_ids, seq_len):
        # shape: text_ids, (batch_size, seq_len) 
        # --> text_embd, (batch_size, seq_len, embedding_dim) 
        text_embd = self.embedding(text_ids)

        # shape: (batch_size, rnn_out_dim)
        encoded = self.encoder(text_embd, seq_len)

        # go through a dense layer before output
        # shape: hidden_out, (batch_size, hidden_dim)
        hidden_out = self.activation(self.dense(encoded))

        # shape: out_logits, (batch_size, output_dim)
        # note that, since we will use cross entropy as the loss func, 
        # we will later use softmax to compute the loss
        out_logits = self.dense_out(hidden_out)
        return out_logits
