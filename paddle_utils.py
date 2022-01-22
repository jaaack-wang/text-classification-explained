'''
Author: Zhengxiang (Jack) Wang 
GitHub: https://github.com/jaaack-wang 
About: Wrapped functions related to paddle for my repository: 
text-classification-explained (context-specific only)
'''

from paddlenlp.datasets import MapDataset
from paddle.io import BatchSampler, DataLoader
from paddlenlp.data import Pad, Stack, Tuple
from paddlenlp.data import Vocab
from collections.abc import Iterable
import jieba


class TextVectorizer:
     
    def __init__(self, tokenizer=None):
        self.tokenize = tokenizer if tokenizer else jieba.lcut
        self.vocab_to_idx = None
        self.idx_to_vocab = None
        self._V = None
    
    def build_vocab(self, text):
        tokens = list(map(self.tokenize, text))
        self._V = Vocab.build_vocab(tokens, unk_token='[UNK]', pad_token='[PAD]')
        self.vocab_to_idx = self._V.token_to_idx
        self.idx_to_vocab = self._V.idx_to_token
        
        print('Two vocabulary dictionaries have been built!\n' \
             + 'Please call \033[1mX.vocab_to_idx | X.idx_to_vocab\033[0m to find out more' \
             + ' where [X] stands for the name you used for this TextVectorizer class.')
        
    def text_encoder(self, text):
        if isinstance(text, list):
            return [self(t) for t in text]
        
        tks = self.tokenize(text)
        out = [self.vocab_to_idx[tk] for tk in tks]
        return out
            
    def text_decoder(self, text_ids, sep=" "):
        if all(isinstance(ids, Iterable) for ids in text_ids):
            return [self.text_decoder(ids, sep) for ids in text_ids]
            
        out = []
        for text_id in text_ids:
            out.append(self.idx_to_vocab[text_id])
            
        return f'{sep}'.join(out)
    
    def __call__(self, text):
        if self.vocab_to_idx:
            return self.text_encoder(text)
        raise ValueError("No vocab is built!")


def example_converter(example, text_encoder, include_seq_len):
    
    text, label = example
    encoded = text_encoder(text)
    if include_seq_len:
        text_len = len(encoded)
        return encoded, text_len, label
    return encoded, label


def get_trans_fn(text_encoder, include_seq_len):
    return lambda ex: example_converter(ex, text_encoder, include_seq_len)


def get_batchify_fn(include_seq_len):
    
    if include_seq_len:
        stack = [Stack(dtype="int64")] * 2
    else:
        stack = [Stack(dtype="int64")]
    
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=0),  
        *stack
    ): fn(samples)
    
    return batchify_fn


def create_dataloader(dataset, 
                      trans_fn, 
                      batchify_fn, 
                      batch_size=64, 
                      shuffle=True, 
                      sampler=BatchSampler):
    
    
    if not isinstance(dataset, MapDataset):
        dataset = MapDataset(dataset)
        
    dataset.map(trans_fn)
    batch_sampler = sampler(dataset, 
                            shuffle=shuffle, 
                            batch_size=batch_size)
    
    dataloder = DataLoader(dataset, 
                           batch_sampler=batch_sampler, 
                           collate_fn=batchify_fn)
    
    return dataloder
