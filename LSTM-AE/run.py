import argparse, os 
from utils import * 
from model import * 

from torch import nn, optim 

parser = argparse.ArgumentParser()

parser.add_argument('--input_dim', default=4096, type=int)
parser.add_argument('--hidden_dim', default=2048, type=int)
parser.add_argument('--output_dim', default=4096, type=int)
parser.add_argument('--num_layers', default=2, type=int)
parser.add_argument('--lr', default=5e-4, type=float)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--device', default='mps', type=str)
parser.add_argument('--num_epochs', default=50, type=int)

args = parser.parse_args()

if __name__ == '__main__':
    
    train_loader = get_loader(args, type='train')
    test_loader = get_loader(args, type='test')
    
    model = Seq2Seq(args).to(args.device)
    
    criterion = nn.MSELoss().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    train_losses, test_losses = [], []
    best_loss = float('inf')
    for epoch in tqdm(range(args.num_epochs)):
        train_loss = train(args, model, train_loader, criterion, optimizer)
        test_loss = evalaute(args, model, test_loader, criterion)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        SAVE_PATH = './model_parameters'
        if best_loss > test_loss:
            best_loss = test_loss 
            
            if not os.path.exists(SAVE_PATH):
                os.mkdir(SAVE_PATH)
            
            torch.save(model.state_dict(), 
                       os.path.join(SAVE_PATH, f'{model.__class__.__name__}-h={args.hidden_dim}-\
                           layer={args.num_layers}-lr={args.lr}.pt'))
            