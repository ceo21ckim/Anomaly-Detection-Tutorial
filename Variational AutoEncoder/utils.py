import os 
import torch 
import torch.nn.functional as F 

from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader 

import time
from tqdm import tqdm 

def criterion(x, x_hat, mean, var):
    reproduction_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1 + var - mean.pow(2) - var.exp())
    return reproduction_loss + KLD 

def elapsed_time(start, end):
    elapsed = end - start 
    elapsed_min = elapsed // 60 
    elapsed_sec = round(elapsed - elapsed_min * 60, 2)
    return elapsed_min, elapsed_sec 


def get_device(methods):
    if methods == 'mps':
        return 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    elif methods == 'cuda':
        return 'cuda' if torch.cuda.is_available() else 'cpu'


def training(args, model, data_loader, criterion, optimizer):
    model.train()
    best_loss = float('inf')

    for epoch in tqdm(range(args.num_epochs), desc='VAE training...'):
        start = time.time()
        total_loss = 0.0
        for x, label in data_loader:
            x = torch.flatten(x, start_dim=1)# flatten images 
            x = x.to(args.device)
            
            optimizer.zero_grad()
            x_hat, mean, var = model(x)
            loss = criterion(x, x_hat, mean, var)
            
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
        if best_loss > total_loss:
            best_loss = total_loss
            save_path = './model_parameters'
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            torch.save(model.state_dict(), 
                        os.path.join(save_path, 
                                    f'{model.__class__.__name__}-lr{args.lr}-batch{args.batch_size}-h{args.hidden_dim}-lt{args.latent_dim}.pt'
                                    )
                        )
        
        end = time.time()
        elapsed_min, elapsed_sec = elapsed_time(start, end )
        print(f'Epoch: [{epoch + 1}/{args.num_epochs}]\tElapsed Time: {elapsed_min}m {elapsed_sec:.2f}s')
        print(f'Average Loss: {total_loss / len(data_loader.dataset):.4f}')

mnist_transform = transforms.Compose([
    transforms.ToTensor()
])

def get_loader(args, type='train'):
    DATA_DIR = '~/data'
    d_kwargs = {'root':DATA_DIR, 'transform':mnist_transform,'download': True}
    if type == 'train':
        d_kwargs['train'] = True
        d_sets = MNIST(**d_kwargs)
        return DataLoader(dataset=d_sets, batch_size=args.batch_size, shuffle=True)
    
    elif type == 'test':
        d_kwargs['train'] = False 
        d_sets = MNIST(**d_kwargs)
        return DataLoader(dataset=d_sets, batch_size=args.batch_size, shuffle=False)