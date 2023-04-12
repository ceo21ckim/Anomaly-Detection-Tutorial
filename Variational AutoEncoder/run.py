from utils import * 
from model import *

import argparse
from torchvision import transforms 
from torchvision.datasets import MNIST

from torch.utils.data import DataLoader 
from torch import optim 

parser = argparse.ArgumentParser(description='parameters for variational autoencoder (VAE)')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--input_dim', default=784, type=int)
parser.add_argument('--hidden_dim', default=400, type=int)
parser.add_argument('-latent_dim', default=200, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--num_epochs', default=30, type=int)
parser.add_argument('--device', default='cpu', type=str)

args = parser.parse_args()




if __name__ == '__main__':
    
    DATA_DIR = '~/data'
    
    args.device = get_device(args.device)
    
    train_loader = get_loader(args, type='train')

    encoder = Encoder(input_dim=args.input_dim, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim)
    decoder = Decoder(latent_dim=args.latent_dim, hidden_dim=args.hidden_dim, output_dim=args.input_dim)
    
    model = VAE(args, encoder, decoder).to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    
    training(args, model, train_loader, criterion, optimizer)
    