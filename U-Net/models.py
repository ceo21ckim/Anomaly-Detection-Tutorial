import torch 
from torch import nn 

class UNet(nn.Module):
    def __init__(self, args):
        super(UNet, self).__init__()
        
        self.C, self.H, self.W = args.n_channels, args.height, args.width 
        self.dim = args.hidden_dim
        
        self.inc = self.DoubleConv(self.C, self.dim)
        
        self.encoder = nn.ModuleList([
            self.downsample(self.dim, self.dim*2), 
            self.downsample(self.dim*2, self.dim*4), 
            self.downsample(self.dim*4, self.dim*8), 
            self.downsample(self.dim*8, self.dim*16)
        ])
        
        self.upscaler = nn.ConvTranspose2d(self.dim*16, self.dim*8, 2, 2)
        
        self.decoder = nn.ModuleList([
            self.upsample(self.dim*16, self.dim*8), 
            self.upsample(self.dim*8, self.dim*4), 
            self.upsample(self.dim*4, self.dim*2)
        ])
        
        self.dec = nn.Sequential(
            self.DoubleConv(self.dim*2, self.dim), 
            nn.Conv2d(self.dim, self.C, 1, 1)
        )
        
        
    def DoubleConv(self, in_c, out_c, k=3, s=1, p=1):
        dc_seqs = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p), 
            nn.BatchNorm2d(out_c), 
            nn.ReLU(), 
            
            nn.Conv2d(out_c, out_c, kernel_size=k, stride=s, padding=p), 
            nn.BatchNorm2d(out_c), 
            nn.ReLU()
        )
        
        return dc_seqs 
    
    def downsample(self, in_c, out_c):
        down_conv = nn.Sequential(
            nn.MaxPool2d(2, 2), 
            self.DoubleConv(in_c, out_c)
        )
        return down_conv
    
    def upsample(self, in_c, out_c): 
        up_conv = nn.Sequential(
            self.DoubleConv(in_c, out_c),
            nn.ConvTranspose2d(out_c, out_c//2, 2, 2), 
        )
        return up_conv 
    
    
    def forward(self, x):
        x = self.inc(x)
        out_list = [x]
        for down in self.encoder:
            out = down(x)
            out_list.append(out)
            x = out
        
        out = self.upscaler(out)
        for i, up in enumerate(self.decoder):
            out = torch.cat([out_list[-(i+2)], out], dim=1)
            out = up(out)
            
        out = torch.cat([out_list[0], out], dim=1) # final Layers
        out = self.dec(out)
        
        return out 
