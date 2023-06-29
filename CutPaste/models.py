import torch
import torch.nn as nn 
import torch.nn.functional as F

from torchvision.models import resnet18



class ProjectionNet(nn.Module):
    def __init__(self, pretrained=True, head_layers=[512]*8 + [128], num_classes=2):
        super(ProjectionNet, self).__init__()
        
        self.resnet18 = resnet(pretrained=pretrained)
        
        last_layer = 512
        sequential_layers = []
        
        for num_neurons in head_layers:
            sequential_layers.append(Block(last_layer, num_neurons))
            last_layer = num_neurons
        
        head = nn.Sequential(
            *sequential_layers
        )
        
        self.resnet18.fc = nn.Identity()
        self.head = head
        self.out = nn.Linear(last_layer, num_classes)
        
    def forward(self, x):
        outs = self.resnet18(x)
        tmp = self.head(outs)
        
        logits = self.out(tmp)
        
        return embeds, logits 
        
    
    def freeze_resnet(self):
        for param in self.resent18.parameters():
            param.requires_grad = False
        
        for param in self.resnet18.parameters():
            param.required_grad = True 
            
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True 
    
    
    def Block(self, last_layer, num_neurons):
        layers = nn.Sequential(
            nn.Linear(last_layer, num_neurons), 
            nn.BatchNorm1d(num_neurons), 
            nn.ReLU(inplace=True)
        )
        return layers 
