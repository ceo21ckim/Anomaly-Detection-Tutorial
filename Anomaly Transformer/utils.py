import pandas as pd 
from torch.utils.data import DataLoader, Dataset 
from sklearn.preprocessing import StandardScaler

def data_preprocess(mode='train'):
    scaler = StandardScaler()
    x_train = pd.read_csv('./data/train.csv').values[:, 1:]
    x_test = pd.read_csv('./data/test.csv').values[:, 1:]
    y_test = pd.read_csv('./data/test_label.csv').values[:, 1:]

    scaler.fit(x_train)
    scaled_x_train = scaler.transform(x_train)
    scaled_x_test = scaler.transform(x_test)
    
    if mode == 'train':
        return scaled_x_train, y_test 
    
    elif mode == 'test':
        return scaled_x_test, y_test 
    
    else:
        raise ValueError('mode must be train or test')


class ATDataset(object):
    def __init__(self, d_set, window_size):
        self.x, self.y = d_set        
        self.window_size = window_size 
        
    def __len__(self):
        return (self.x.shape[0] - self.window_size) + 1
    
    def __getitem__(self, idx):
        x = self.x[idx:idx + self.window_size]
        y = self.y[idx:idx + self.window_size]
        return (x, y)
    
def get_loader(args, mode='train'):
    inputs, labels = data_preprocess(mode)
    d_set = ATDataset([inputs, labels], args.window_size)
    shuffle = False
    if mode == 'train':
        shuffle = True 
    
    return DataLoader(d_set, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers)