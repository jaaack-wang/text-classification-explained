'''
Author: Zhengxiang (Jack) Wang 
GitHub: https://github.com/jaaack-wang 
About: Utility functions for training and evaluating PyTorch models
for my repository: text-classification-explained (context-specific only)
'''
import torch
import torch.nn.functional as F
import jieba
from torchtext.vocab import build_vocab_from_iterator
from collections import defaultdict
from collections.abc import Iterable
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import Dataset, DataLoader


def to_tensor(dataset):
    for i, batch in enumerate(dataset):
        for j, e in enumerate(batch):
            dataset[i][j] = torch.tensor(e)
            
    return dataset


class PyTorchUtils:
    
    def __init__(self, model, optimizer, 
                 criterion, include_seq_len):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.include_seq_len = include_seq_len
        
    @staticmethod
    def _accuracy(preds, y):
        preds = torch.round(torch.sigmoid(preds))
        correct = sum((torch.all(preds[i]==y[i]) 
                       for i in range(y.shape[0]))).float()
    
        accu = correct / y.shape[0]
        return accu

    
    def _train(self, dataset, one_hot_y=True):
        
        if one_hot_y:
            convert = lambda x: F.one_hot(x)
        else:
            convert = lambda x: x
        
        epoch_loss, epoch_accu = 0, 0
        self.model.train()
    
        for batch in dataset:
            self.optimizer.zero_grad()
        
            if self.include_seq_len:
                preds = self.model(batch[0], batch[1])
            else:
                preds = self.model(batch[0])
                
            loss = self.criterion(preds, convert(batch[-1]).float())
            accu = self._accuracy(preds, convert(batch[-1]))
    
            loss.backward()
            self.optimizer.step()
        
            epoch_loss += loss.item()
            epoch_accu += accu.item()
        
        return {f"Train loss": f"{epoch_loss/len(dataset):.5f}", 
                f"Train accu": f"{epoch_accu/len(dataset)*100:.2f}"}
     
    def evaluate(self, dataset, eval_subj="Test", one_hot_y=True):
        
        if one_hot_y:
            convert = lambda x: F.one_hot(x)
        else:
            convert = lambda x: x
    
        epoch_loss, epoch_accu = 0, 0
        self.model.eval()
    
        with torch.no_grad():
            for batch in dataset:

                if self.include_seq_len:
                    preds = self.model(batch[0], batch[1])
                else:
                    preds = self.model(batch[0])
            
                loss = self.criterion(preds, convert(batch[-1]).float())
                accu = self._accuracy(preds, convert(batch[-1]))

                epoch_loss += loss.item()
                epoch_accu += accu.item()
        
        return {f"{eval_subj} loss": f"{epoch_loss/len(dataset):.5f}", 
                f"{eval_subj} accu": f"{epoch_accu/len(dataset)*100:.2f}"}
    
    def train(self, train_set, dev_set=None, epochs=1, 
              one_hot_y=True, save_model=False):
        
        for idx in range(epochs):
            train_res = self._train(train_set, one_hot_y)
            print(f"Epoch {idx+1}/{epochs}", train_res)
            
            if dev_set:
                dev_res = self.evaluate(dev_set, "Dev", one_hot_y)
                print(f"Validation...", dev_res)
            print()
            
        if save_model:
            torch.save(self.model.state_dict(), 'model.pt')
            print(f"Model has been saved in ./model.pt")


class TextVectorizer:
     
    def __init__(self, tokenizer=None):
        self.tokenize = tokenizer if tokenizer else jieba.lcut
        self.vocab_to_idx = {}
        self.idx_to_vocab = {}
        self._V = None
    
    def build_vocab(self, text):
        tokens = list(map(self.tokenize, text))
        
        self._V = build_vocab_from_iterator(tokens, specials=['[PAD]', '[UNK]'])
        for idx, tk in enumerate(self._V.get_itos()):
            self.vocab_to_idx[tk] = idx
            self.idx_to_vocab[idx] = tk
        
        self.vocab_to_idx = defaultdict(lambda: self.vocab_to_idx['[UNK]'], 
                                        self.vocab_to_idx)
        
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


def _batchify_fn(batch, 
                 text_encoder, 
                 pad_idx=0,
                 max_seq_len=None, 
                 include_seq_len=False, 
                 dtype=torch.int64):
    
    # ----- pad func for a list -----
    def _pad(lst, max_len):
        dif = max_len - len(lst)
        if dif > 0:
            return lst + [pad_idx] * dif
        if dif < 0:
            return lst[:max_len]
        return lst
    
    # ----- pad func for a bacth of lists -----
    def pad(data):
        if max_seq_len:
            max_len = max_seq_len
        else:
            max_len = max(len(d) for d in data)
        
        for i, d in enumerate(data):
            data[i] = _pad(d, max_len)
        
        return data
    
    # ----- turn a list of int into tensor -----
    def to_tensor(x):
        return torch.tensor(x, dtype=dtype)
    
    
    # ----- start batchifying -----
    out_t, out_l = [], []
    
    for (text, label) in batch:
        out_t.append(text_encoder(text)) 
        out_l.append(label)
    
    if include_seq_len:
    
    # if include_seq_len and max_seq_len, longer text will be trimmed
    # hence, their text_seq_len is also reduced to the max_seq_len
        if max_seq_len:
            t_len = to_tensor([len(t) if len(t) < max_seq_len 
                               else max_seq_len for t in out_t])
        else:
            t_len = to_tensor([len(t) for t in out_t])
        
    # torch.cat put a list of tensor together
    out_t = to_tensor(pad(out_t))
    out_l = to_tensor(out_l)
    
    if include_seq_len:
        return out_t, t_len, out_l
    
    return out_t, out_l


def get_batchify_fn(text_encoder, 
                    pad_idx=0,
                    max_seq_len=None, 
                    include_seq_len=False, 
                    dtype=torch.int64):
    
    return lambda ex: _batchify_fn(ex, text_encoder, 
                                   pad_idx, max_seq_len, 
                                   include_seq_len, dtype)

def create_dataloader(dataset, 
                      batchify_fn, 
                      batch_size=64, 
                      shuffle=True):
    
    
    if not isinstance(dataset, Dataset):
        dataset = to_map_style_dataset(dataset)
        
    
    dataloder = DataLoader(dataset, 
                           batch_size=batch_size, 
                           shuffle=shuffle,
                           collate_fn=batchify_fn)
    
    return dataloder
