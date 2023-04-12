from torch import nn 
import torch 

class Encoder(nn.Module):
    def __init__(self, input_dim=64*64, hidden_dim=1024, num_layers=2):
        super(Encoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers 
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1, bidirectional=False)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        o, (h, c) = self.lstm(x)
        
        return (h, c)
    
class Decoder(nn.Module):
    def __init__(self, input_dim=64*64, hidden_dim=1024, output_dim=64*64, num_layers=2):
        super(Decoder, self).__init__()
        
        self.hidden_dim = hidden_dim 
        self.output_dim = output_dim 
        self.num_layers = num_layers 
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1, bidirectional=False)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, hidden):
        o, (h, c) = self.lstm(x, hidden)
        pred_y = self.fc(o)
        
        return pred_y, (h, c)

class Seq2Seq(nn.Module):
    def __init__(self, args):
        super(Seq2Seq, self).__init__()
        
        hidden_dim = args.hidden_dim
        input_dim = args.input_dim 
        output_dim = args.output_dim 
        num_layers = args.num_layers 
        
        self.encoder = Encoder(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers
        )
        
        self.reconstruct_decoder = Decoder(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=output_dim, 
            num_layers=num_layers
        ) # reconstruct 
        
        self.predict_decoder = Decoder(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=output_dim, 
            num_layers=num_layers
        ) # predict next time 
        
    def forward(self, source, type='prediction'):
        
        batch_size, seq_len, img_size = source.size() # (B, S, W, H)
        hidden = self.encoder(source)
        
        outs = []
        temp_input = torch.zeros((batch_size, 1, img_size), dtype=torch.float).to(source.device)
        
        if type=='prediction':
            for t in range(seq_len):
                temp_input, hidden = self.predict_decoder(temp_input, hidden)
                outs.append(temp_input)
            
            outs = torch.cat(outs, dim=1)
            
            return outs 
        
        elif type=='reconstruction':
            for t in range(seq_len):
                temp_input, hidden = self.reconstruct_decoder(temp_input, hidden)
                outs.append(temp_input)
            
            outs = torch.cat(outs, dim=1)
            
            return outs 
    
    def generate(self, source):
        batch_size, seq_len, img_size = source.size()
        
        hidden = self.encoder(source)
        
        outs = []
        
        temp_input = torch.zeros((batch_size, 1, img_size), dtype=torch.float).to(source.device)
        
        for t in range(seq_len):
            temp_input, hidden = self.predict_decoder(temp_input, hidden)
            outs.append(temp_input)
            
        return torch.cat(outs, dim=1)
    
    def reconstruct(self, source):
        batch_size, seq_len, img_size = source.size()
        
        hidden = self.encoder(source)
        
        outs = []
        
        temp_input = torch.zeros((batch_size, 1, img_size), dtype=torch.float).to(source.device)
        
        for t in range(seq_len):
            temp_input, hidden = self.reconstruct_decoder(temp_input, hidden)
            outs.append(temp_input)
            
        return torch.cat(outs, dim=1)
    
    