'''
Author: Zhengxiang (Jack) Wang 
GitHub: https://github.com/jaaack-wang 
About: Simple GRU model for text classification using tensorflow. 
'''
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


class GRU(keras.Model):
    
    def __init__(self,
                 vocab_size,
                 output_dim, 
                 embedding_dim=100,
                 rnn_hidden_dim=128,
                 mask_zero=True, 
                 hidden_dim_out=50,
                 n_layers=1,
                 bidirectional=False,
                 dropout_rate=0.0,
                 activation=layers.ReLU(), 
                 activation_out=tf.sigmoid):
        
        super().__init__()
        
        self.embedding = layers.Embedding(
            vocab_size, embedding_dim, mask_zero=mask_zero)

        assert isinstance(n_layers, int) and n_layers >= 1, "" \
        + "n_layers must be an integer greater than 0."
        
        self.n_layers = n_layers
        
        def get_gru(bi_direct, return_seq):
            if bi_direct:
                return layers.Bidirectional(
                    layers.GRU(rnn_hidden_dim, 
                                     return_sequences=return_seq))
            
            return layers.GRU(rnn_hidden_dim, 
                                    return_sequences=return_seq)
  
        if self.n_layers == 1:
            self.gru = get_gru(bidirectional, False)
        else:
            self.gru = []
            for _ in range(self.n_layers-1):
                self.gru.append(get_gru(bidirectional, True))
            self.gru.append(get_gru(bidirectional, False))
        
        self.dense = layers.Dense(hidden_dim_out)
        self.activation = activation
        self.dense_out = layers.Dense(output_dim)
        self.activation_out = activation_out
    
    def encoder(self, embd):
        if self.n_layers == 1:
            return self.gru(embd)

        for gru in self.gru:
            embd = gru(embd)
        return embd

    def call(self, text_ids):
        
        # shape: text_ids, (batch_size, text_seq_len) 
        # --> text_ids_embd, (batch_size, text_seq_len, embedding_dim) 
        text_embd = self.embedding(text_ids)
        
        # shape: encoded, (batch_size, rnn_hidden_dim)
        encoded = self.encoder(text_embd)

        # go through a dense layer before output
        # shape: hidden_out, (batch_size, hidden_dim)
        hidden_out = self.activation(self.dense(encoded))

        # shape: out_logits, (batch_size, output_dim)
        out_logits = self.dense_out(hidden_out)
        
        # for binary, use "tf.sigmoid" as the output activation func; 
        # for multi-class, use "tf.math.softmax" instead, and
        # you also need to one-hot encode your labels as well
        return self.activation_out(out_logits)
