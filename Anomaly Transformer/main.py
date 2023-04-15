import torch 
from torch import nn, optim

from model import * 
from utils import * 

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--window_size', default=60, type=int)
parser.add_argument('--input_c', default=25, type=int)
parser.add_argument('--output_c', default=25, type=int)
parser.add_argument('--num_layers', default=3)
parser.add_argument('--num_epochs', default=200)
parser.add_argument('--batch_size', default=512)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--lambda_', default=3)

args = parser.parse_args()

args.SAVE_DIR = os.path.join(os.getcwd(), 'model_parameters')

if __name__ == '__main__':
    train_loader = get_loader(args, mode='train')
    valid_loader = get_loader(args, mode='test')
    
    model = AnomalyTransformer(
        window_size = args.window_size, 
        enc_in = args.input_c, 
        c_out = args.output_c, 
        e_layers = args.num_layers
    ).to(args.device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    rec_loss = nn.MSELoss().to(args.device)
    
    ## training
    train(args, model, train_loader, valid_loader, [rec_loss, association_discrepancy], optimizer)
    