import argparse

from utils import * 
from data_utils import * 
from models import * 

from torch import optim 


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--d_name', default='yelp')
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--homo', default=1, type=int)
    parser.add_argument('-tr', '--train_ratio', default=0.4, type=float)
    parser.add_argument('-o', '--order', default=2, type=int)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--h_feats', default=64, type=int)

    args = parser.parse_args()
    
    graph = Dataset(args.d_name, args.homo).graph
    args.in_feats = graph.ndata['feature'].shape[1] # hyperspectral channels
    model = BWGNN(args.in_feats, args.h_feats, args.num_classes, graph, d=args.order)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    final_tmf1, final_tauc = train(args, model, graph, optimizer)
