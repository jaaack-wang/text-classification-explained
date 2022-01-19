'''
Author: Zhengxiang (Jack) Wang 
GitHub: https://github.com/jaaack-wang 
About: Utility functions for training and evaluating PyTorch models
for my repository: text-matching-explained (context-specific only)
'''
import torch
import torch.nn.functional as F


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
