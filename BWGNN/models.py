from torch import nn

from utils import * 

class BWGNN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, graph, d=2, batch=False):
        super(BWGNN, self).__init__()
        
        self.g = graph
        self.thetas = calculate_theta2(d=d)
        self.conv = []
        
        for i in range(len(self.thetas)):
            if not batch:
                self.conv.append(PolyConv(h_feats, h_feats, self.thetas[i]))
            
            else:
                self.conv.append(PolyConvBatch(h_feats, h_feats, self.thetas[i]))
        
        self.fc1 = nn.Linear(in_feats, h_feats)
        self.fc2 = nn.Linear(h_feats, h_feats)
        self.fc3 = nn.Linear(h_feats*len(self.conv), h_feats)
        self.fc4 = nn.Linear(h_feats, num_classes)
        self.act = nn.ReLU()
        self.d = d
        
    
    def forward(self, in_feat, g=None, mode='train'):
        h = self.fc1(in_feat)
        h = self.act(h)
        h = self.fc2(h)
        h = self.act(h)
        h_final = torch.zeros([len(in_feat), 0])
        
        for conv in self.conv:
            if mode == 'train':
                h0 = conv(self.g, h)
            
            elif mode == 'test':
                h0 = conv(g, h)
            
            elif mode == 'batch':
                h0 = conv(g[0], h)
            h_final = torch.cat([h_final, h0], -1)
        h = self.fc3(h_final)
        h = self.act(h)
        h = self.fc4(h)
        return h 
    
class BWGNN_Hetero(nn.Module):
    def __init__(self, in_feats, out_feats, num_classes, graph, d=2):
        super(BWGNN_Hetero, self).__init__()
        self.g = graph 
        self.thetas = calculate_theta2(d=d)
        self.h_feats = h_feats 
        self.conv = [PolyConv(h_feats, h_feats, theta) for theta in self.thetas]
        self.fc1 = nn.Linear(in_feats, h_feats)
        self.fc2 = nn.Linear(h_feats, h_feats)
        self.fc3 = nn.Linear(h_feats*len(self.conv), h_feats)
        self.fc4 = nn.Linear(h_feats, num_classes)
        self.act = nn.LeakyReLU()

    def forward(self, in_feat):
        h = self.fc1(in_feat)
        h = self.act(h)
        h = self.fc2(h)
        h = self.act(h)
        h_all = []

        for relation in self.g.canonical_etypes:
            h_final = torch.zeros([len(in_feat), 0])
            for conv in self.conv:
                h0 = conv(self.g[relation], h)
                h_final = torch.cat([h_final, h0], -1)

            h = self.fc3(h_final)
            h_all.append(h)
        h_all = torch.stack(h_all).sum(0)
        h_all = self.act(h_all)
        h_all = self.fc4(h_all)
        return h_all 
    


class PolyConv(nn.Module):
    def __init__(self, in_feats, out_feats, theta):
        super(PolyConv, self).__init__()
        self._theta = theta
        self._k = len(self._theta)
        self._in_feats = in_feats 
        self._out_feats = out_feats 
        self.activation = nn.LeakyReLU() 
        self.linear = nn.Linear(in_feats, out_feats)
        self._init_weight()
        
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, graph, feat):
        with graph.local_scope():
            D_invsqrt = torch.pow(graph.in_degrees().float().clamp(min=1), -0.5).unsqueeze(-1).to(feat.device)
            h = self._theta[0] * feat
            
            for k in range(1, self._k):
                feat = unnLaplacian(feat, D_invsqrt, graph)
                h += self._theta[k]*feat 
        
        return h 
    
    
class PolyConvBatch(nn.Module):
    def __init__(self, in_feats, out_feats, theta):
        super(PolyConvBatch, self).__init__()
        self._theta = theta 
        self._k = len(self._theta)
        self._in_feats = in_feats 
        self._out_feats = out_feats 
        self.activation = nn.LeakyReLU()
        self._init_weight()
        
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, block, feat):
        with block.local_scope():
            D_invsqrt = torch.pow(block.out_degrees().float().clamp(min=1), -0.5).unsqueeze(-1).to(feat.device)
            h = self._theta[0] * feat 
            for k in range(1, self._k):
                feat = unnLaplacian(feat, D_invsqrt, block)
                h += self._theta[k]*feat 
        
        return h
