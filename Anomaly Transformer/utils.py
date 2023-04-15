import pandas as pd 
import numpy as np 
from torch.utils.data import DataLoader 
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm 

import torch 
import os 

def data_preprocess(mode='train'):
    scaler = StandardScaler()
    x_train = pd.read_csv('./data/train.csv').values[:, 1:]
    x_test = pd.read_csv('./data/test.csv').values[:, 1:]
    y_test = pd.read_csv('./data/test_label.csv').values[:, 1:]
    
    
    print(f'train shape: {x_train.shape}\t test shape: {x_test.shape}')
    
    x_train = np.nan_to_num(x_train)
    x_test = np.nan_to_num(x_test)
    
    scaler.fit(x_train)
    scaled_x_train = scaler.transform(x_train)
    scaled_x_test = scaler.transform(x_test)
    
    if mode == 'train':
        return scaled_x_train, y_test 
    
    elif mode == 'test':
        return scaled_x_test, y_test 
    
    else:
        raise ValueError('mode must be train or test')


class ATDataset(object):
    def __init__(self, d_set, window_size):
        self.x, self.y = d_set        
        self.window_size = window_size 
        
    def __len__(self):
        return (self.x.shape[0] - self.window_size) + 1
    
    def __getitem__(self, idx):
        x = self.x[idx:idx + self.window_size]
        y = self.y[idx:idx + self.window_size]
        return (x, y)
    
def get_loader(args, mode='train'):
    inputs, labels = data_preprocess(mode)
    d_set = ATDataset([inputs, labels], args.window_size)
    shuffle = False
    if mode == 'train':
        shuffle = True 
    
    return DataLoader(d_set, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers)






## Metrics ...

def kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def association_discrepancy(args, series, prior):
    prior_association = prior / torch.unsqueeze(torch.sum(prior, dim=-1), dim=-1).repeat(1, 1, 1, args.window_size)
    series_association = series
    
    left = torch.mean(
        kl_loss(prior_association, series_association)
    )

    right = torch.mean(
        kl_loss(series_association, prior_association)
    )
        
    return left + right 


### Training...

def evaluate(args, model, valid_loader, criterion):
    
    model.eval()
    
    rec_loss, association_discrepancy = criterion
    
    min_losses, max_losses = [], []
    
    iterator = tqdm(valid_loader, desc='evaluating...')
    
    for batch in iterator:
        x, label = tuple(b.to(args.device) for b in batch)
        x = x.float()
        
        series_loss, prior_loss = 0.0, 0.0
        
        x_hat, series, prior, _ = model(x)
        
        for i in range(len(prior)):
            series_loss += association_discrepancy(args, series[i], prior[i].detach())
            prior_loss += association_discrepancy(args, series[i].detach(), prior[i])

        series_loss = series_loss / len(prior)
        prior_loss = prior_loss / len(prior)
        reconstruction_loss = rec_loss(x, x_hat)
        
        
        max_loss = reconstruction_loss - args.lambda_ * prior_loss
        min_loss = reconstruction_loss + args.lambda_ * series_loss 
        
        iterator.set_postfix({
            'max_loss':max_loss.item(), 
            'min_loss':min_loss.item(), 
            'rec_loss':reconstruction_loss.item()
        }
        )
        
        
    max_losses.append(max_loss.item())
    min_losses.append(min_loss.item())
    
    return np.average(min_losses), np.average(max_losses)


def train(args, model, train_loader, valid_loader, criterion:list, optimizer):
    '''
    input:
        args: args
        model: Anomaly Transformer
        train_loader, valid_loader: torch.utils.data.DataLoader
        criterion: [criterion1, criterion2]
                    creterion1: reconstruction loss e.i., MSELoss, MAELoss, etc.
                    criterion2: association discrepancy
                    
        optimizer: optimization function. Adam
    '''
    
    rec_loss, association_discrepancy = criterion
    counts = 0 
    for epoch in range(args.num_epochs):
        min_best_loss = float('inf')
        max_best_loss = float('-inf')
        
        model.train()
        iterator = tqdm(train_loader, desc=f'Epoch: [{epoch}/{args.num_epochs}]')
        
        for batch in iterator:
            train_losses = 0.0
            x, label = tuple(b.to(args.device) for b in batch)
            
            optimizer.zero_grad()
            x = x.float()
            
            x_hat, series, prior, sigma = model(x)
            
            series_loss, prior_loss = 0.0, 0.0
            
            for i in range(len(prior)):
                series_loss += association_discrepancy(args, series[i], prior[i].detach())
                prior_loss += association_discrepancy(args, series[i].detach(), prior[i])
                
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)
            
            reconstruction_loss = rec_loss(x, x_hat)
            
            max_loss = reconstruction_loss - args.lambda_ * prior_loss 
            min_loss = reconstruction_loss + args.lambda_ * series_loss 
            
            iterator.set_postfix({
                'min_loss':min_loss.item(), 
                'max_loss':max_loss.item(), 
                'rec_loss':reconstruction_loss.item()})
            
            max_loss.backward(retain_graph=True)
            min_loss.backward()
            optimizer.step()
        
        min_valid_loss, max_valid_loss = evaluate(args, model, valid_loader, criterion)
        if min_best_loss > min_valid_loss : 
            min_best_loss = min_valid_loss
            if not os.path.exists(args.SAVE_DIR):
                os.mkdir(args.SAVE_DIR)

            torch.save(model.state_dict(), f'{args.SAVE_DIR}/Anomaly-Transformer.pt')

        if max_best_loss < max_valid_loss :
            max_best_loss = max_valid_loss 
            if not os.path.exists(args.SAVE_DIR):
                os.mkdir(args.SAVE_DIR)

            torch.save(model.state_dict(), f'{args.SAVE_DIR}/Anomaly-Transformer.pt')

        if min_best_loss < min_valid_loss and max_best_loss > max_valid_loss : 
            counts += 1
            if counts >= 3:
                break 