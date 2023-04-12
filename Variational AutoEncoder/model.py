from utils import * 


import argparse, time, os
import numpy as np 
from tqdm import tqdm 

import torch 
from torch import nn, optim 
import torch.nn.functional as F 


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        
        self.fc_input   = nn.Linear(input_dim, hidden_dim)
        self.fc_input2  = nn.Linear(hidden_dim, hidden_dim)
        self.mean       = nn.Linear(hidden_dim, latent_dim)
        self.var        = nn.Linear(hidden_dim, latent_dim)
        
        self.LeakyReLU  = nn.LeakyReLU(0.2)
        
        self._init_weight()
        
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        h_      = self.LeakyReLU(self.fc_input(x))
        h_      = self.LeakyReLU(self.fc_input2(h_))
        mean    = self.mean(h_)
        log_var = self.var(h_)
        
        return mean, log_var 

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        
        self.fc_hidden  = nn.Linear(latent_dim, hidden_dim)
        self.fc_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out     = nn.Linear(hidden_dim, output_dim)
        
        self.LeakyReLU  = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        h = self.LeakyReLU(self.fc_hidden(x))
        h = self.LeakyReLU(self.fc_hidden2(h))
        
        outs = self.sigmoid(self.fc_out(h))
        return outs 

class VAE(nn.Module):
    def __init__(self, args, Encoder, Decoder):
        super(VAE, self).__init__()
        self.args = args
        self.Encoder = Encoder 
        self.Decoder = Decoder 
        
    def reparameterization(self, mean, var):
        eps = torch.randn_like(var).to(self.args.device)
        z = mean + var*eps 
        return z 
    
    def forward(self, x):
        mean, var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * var))
        outs = self.Decoder(z)
        return outs, mean, var 