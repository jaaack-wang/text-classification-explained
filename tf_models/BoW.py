'''
Author: Zhengxiang (Jack) Wang 
GitHub: https://github.com/jaaack-wang 
About: Simple BoW model for text classification using tensorflow. 
'''
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


class BoW(keras.Model):
    
    def __init__(self, 
                 vocab_size,
                 output_dim, 
                 embedding_dim=100, 
                 mask_zero=True, 
                 hidden_dim=50, 
                 activation=layers.ReLU(), 
                 activation_out=tf.sigmoid):
        
        super().__init__()
        self.embedding = layers.Embedding(
            vocab_size, embedding_dim, mask_zero=mask_zero)
        self.dense = layers.Dense(hidden_dim)
        self.activation = activation
        self.dense_out = layers.Dense(output_dim)
        self.activation_out = activation_out
    
    def encoder(self, embd):
        return tf.math.reduce_sum(embd, axis=1)
    
    def call(self, text_ids):
                
        # shape: text_ids, (batch_size, text_seq_len) 
        # --> text_embd, (batch_size, text_seq_len, embedding_dim) 
        text_embd = self.embedding(text_ids)

        # shape: encoded, (batch_size, embedding_dim)
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
