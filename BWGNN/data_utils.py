from dgl.data import FraudAmazonDataset, FraudYelpDataset

import dgl

class Dataset:
    def __init__(self, name='yelp', homo=True):
        if name == 'yelp':
            dataset = FraudYelpDataset()
            graph = dataset[0]
            
        elif name == 'amazon':
            dataset = FraudAmazonDataset()
            graph = dataset[0]    
                
        if homo:
            graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
            graph = dgl.add_self_loop(graph)
        
        graph.ndata['label'] = graph.ndata['label'].long().squeeze(-1)
        graph.ndata['feature'] = graph.ndata['feature'].float()
        print(graph)
        
        self.graph = graph
