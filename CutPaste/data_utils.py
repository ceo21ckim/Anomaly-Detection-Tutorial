import os

from PIL import Image

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from cutpaste import CutPasteNormal, CutPaste, CutPasteScar, cut_paste_collate_fn

class Repeat(Dataset):
    def __init__(self, org_dataset, new_length):
        self.org_dataset = org_dataset
        self.org_length = len(self.org_dataset)
        self.new_length = new_length 
        
    def __len__(self):
        return self.new_length
    
    def __getitem__(self, idx):
        return self.org_dataset[idx % self.org_length]
    
class MVTecData(Dataset):
    def __init__(self, path, size=256, cutpaste_type=CutPasteNormal, transform=None, mode='train'):
        self.path = path 
        self.mode = mode 
        self.size = size
        
        if transform:
            if self.mode == 'train':
                self.cutpaste_transform = transforms.Compose([
                    transforms.ToTensor(), 
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])]
                )

                self.transform = transforms.Compose([
                    transforms.ColorJitter(0.1, 0.1, 0.1, 0.1), 
                    transforms.Resize((self.size, self.size)), 
                    cutpaste_type(self.cutpaste_transform)]
                )
            
            elif self.mode == 'test':
                self.transform = transforms.Compose([
                    transforms.Resize((self.size.self.size)), 
                    transforms.ToTensor(), 
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])]
                )
        
        self.image_names = glob.glob(os.path.join(self.path, 'train/*/*.png'))
        self.imgs = Parallel(n_jobs=10)(delayed(lambda file: Image.open(file).resize((size, size)).convert('RGB'))(file) 
                                        for file in self.image_names)
        
        self.labels = [os.path.split(os.path.split(imgs)[0])[-1] for imgs in self.image_names]
        
    def __len__(self):
        return len(self.imge_names)
    
    def __getitem__(self, idx):
        img = self.imgs[idx].copy()
        label = self.labels[idx]
        
        img = self.transform(img)
        
        return img, label != 'good'


def get_loader(args, path, mode='train'):
    d_set = MVTecData(path, size=args.size, cutpaste_type=args.cutpaste_type, transform=args.transform, mode=mode)
    if mode == 'train':
        return DataLoader(Repeat(d_set, 3000), batch_size=args.batch_size, drop_last=True, 
                         shuffle=True, num_workers=4, collate_fn=cut_paste_collate_fn)
    
    else:
        return DataLoader(d_set, batch_size=64, shuffle=False, num_workers=0)
    

