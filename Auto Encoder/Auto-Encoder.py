import torch 
from torch import nn 

class AutoEncoder(nn.Module):
    def __init__(self, args):
        super(AutoEncoder, self).__init__()
        
        self.C, self.H, self.W = args.n_channels, args.height, args.width 
        self.dim = args.ae_dim
        self.ratio = args.mask_ratio
        
        self.inc = self.DoubleConv(self.C, self.dim)
        
        self.encoder = nn.ModuleList([
            self.downsample(self.dim, self.dim*2), 
            self.downsample(self.dim*2, self.dim*4), 
            self.downsample(self.dim*4, self.dim*8), 
            self.downsample(self.dim*8, self.dim*16)
        ])
        
        self.upscaler = nn.ConvTranspose2d(self.dim*16, self.dim*8, 2, 2)
        
        self.decoder = nn.ModuleList([
            self.upsample(self.dim*8, self.dim*4), 
            self.upsample(self.dim*4, self.dim*2),
            self.upsample(self.dim*2, self.dim)
        ])
        
        self.dec = nn.Sequential(
            self.DoubleConv(self.dim, self.dim//2),
            nn.BatchNorm2d(self.dim//2),
            nn.Conv2d(self.dim//2, self.C, 1, 1), 
            nn.Sigmoid()
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
            nn.ConvTranspose2d(out_c, out_c, 2, 2), 
        )
        return up_conv 
    
    
    def forward(self, x):
        if self.ratio:
            x = self.mask(x)
        x = self.inc(x)
        for down in self.encoder:
            out = down(x)
            x = out
        
        out = self.upscaler(out)
        for i, up in enumerate(self.decoder):
            out = up(out)
            
        out = self.dec(out)
        
        return out
    
    def mask(self, x):
        shape = x.shape
        m = torch.rand(shape).uniform_(0, 1).to(x.device)
        m = torch.where(m < self.ratio, x, 0)

        return x * m
