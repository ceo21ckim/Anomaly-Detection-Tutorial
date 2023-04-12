from tqdm import tqdm 

from torchvision import transforms 
from MovingMNIST import MovingMNIST #https://github.com/tychovdo/MovingMNIST

from torch.utils.data import DataLoader 
import torch 



mnist_transform = transforms.Compose([
    transforms.ToTensor()
])

def get_loader(args, type='train'):
    DATA_DIR = '~/data'
    d_kwargs = {'root':DATA_DIR, 'transform':mnist_transform,'download': True}
    if type == 'train':
        d_kwargs['train'] = True
        d_sets = MovingMNIST(**d_kwargs)
        return DataLoader(dataset=d_sets, batch_size=args.batch_size, shuffle=True)
    
    elif type == 'test':
        d_kwargs['train'] = False 
        d_sets = MovingMNIST(**d_kwargs)
        return DataLoader(dataset=d_sets, batch_size=args.batch_size, shuffle=False)
    
def train(args, model, data_loader, criterion, optimizer):
    train_loss = 0.0
    model.train()
    train_iterator = tqdm(data_loader, total=len(data_loader), desc='training...')
    for batch in train_iterator:
        future_data, past_data = batch
        
        batch_size, example_size = past_data.size(0), past_data.size(1)
        
        past_data = past_data.view(batch_size, example_size, -1).float().to(args.device)
        future_data = future_data.view(batch_size, example_size, -1).float().to(args.device)
        
        recon_outs = model(past_data, 'reconstruction')
        pred_outs = model(past_data, 'prediction')
        
        inv_idx = torch.arange(example_size, -1, -1).long()
        recon_loss = criterion(past_data, recon_outs[:, inv_idx, :])
        pred_loss = criterion(future_data, pred_outs)
        
        final_loss = recon_loss + pred_loss 
        
        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()
        
        train_iterator.set_postfix({
            'train_loss': float(final_loss)
        })
        
        train_loss += final_loss.mean().item()
    train_loss /= len(train_iterator)
    
    return train_loss
            

def evalaute(args, model, data_loader, criterion):
    
    model.eval()
    eval_loss = 0.0
    test_iterator = tqdm(data_loader, total=len(data_loader),desc='testing...' )
    with torch.no_grad():
        for batch in test_iterator:
            future_data, past_data = batch
            
            batch_size, example_size = past_data.size(0), past_data.size(1)
            past_data = past_data.view(batch_size, example_size, -1).float().to(args.device)
            future_data = future_data.view(batch_size, example_size, -1).float().to(args.device)
            
            recon_outs = model(past_data, 'reconstructioin')
            pred_outs = model(past_data, 'prediction')
            
            inv_idx = torch.arange(example_size, -1, -1).long()
            recon_loss = criterion(past_data, recon_outs[:, inv_idx, :])
            pred_loss = criterion(future_data, pred_outs)
            
            final_loss = recon_loss + pred_loss 
            
            eval_loss += final_loss.mean().item()
            
            test_iterator.set_postfix({
                'eval_loss': float(final_loss)
            })
    eval_loss = eval_loss / len(data_loader)
    print(f'Evaluation Score: [{eval_loss}]')
    return eval_loss 