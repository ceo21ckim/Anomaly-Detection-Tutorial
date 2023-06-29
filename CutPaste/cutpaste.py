import random, math

import torch 
from torchvision import transforms 



def cut_paste_collate_fn(batch):
    img_types = list(zip(*batch))
    
    return [torch.stack(imgs) for imgs in img_types]


class CutPaste:
    def __init__(self, colorJitter=0.1, transform=None):
        self.transform = transform 
        
        if colorJitter is None:
            self.colorJitter = None
        
        else:
            self.colorJitter = transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)
        
    def __call__(self, org_img, img):
        if self.transform:
            img = self.transform(img)
            org_img = self.transform(org_img)
        
        return org_img, img
    

class CutPasteNormal(CutPaste):
    def __init__(self, area_ratio=[0.02, 0.15], aspect_ratio=0.3, **kwargs):
        super(CutPasteNormal, self).__init__(**kwargs)
        
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio
        
    def __call__(self, img):
        h = img.size[0]
        w = img.size[1]
        
        ratio_area = random.uniform(self.area_ratio[0], self.area_ratio[1]) * w * h
        
        log_ratio = torch.log(torch.tensor((self.aspect_ratio, 1/self.aspect_ratio)))
        
        aspect = torch.exp(
            torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
        ).item()
        
        cut_w = int(round(math.sqrt(ratio_area * aspect)))
        cut_h = int(round(math.sqrt(ratio_area / aspect)))
        
        from_location_h = int(random.uniform(0, h - cut_h))
        from_location_w = int(random.uniform(0, w - cut_w))
        
        box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
        path = img.crop(box)
        
        if self.clorJitter:
            patch = self.colorJitter(patch)
        
        to_location_h = int(random.uniform(0, h - cut_h))
        to_location_w = int(rnadom.uniform(0, w - cut_w))
        
        insert_box = [to_location_w, to_location_h, to_location_w + cut_w, to_location_h + cut_h]
        augmented = img.copy()
        augmented.paste(patch, insert_box)
        
        return super().__call__(img, augmented)
    

class CutPasteScar(CutPaste):
    def __init__(self, width=[2, 16], height=[10, 25], rotation=[-45, 45], **kwargs):
        super(CutPasteScar, self).__init__(**kwargs)
        self.width = width
        self.height = height
        self.rotation = rotation
        
    def __call__(self, img):
        h = img.size[0]
        w = img.size[1]
        
        cut_w = random.uniform(*self.width)
        cut_h = random.uniform(*self.height)
        
        from_location_h = int(random.uniform(0, h - cut_h))
        from_location_w = int(random.uniform(0, w - cut_w))
        
        box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
        patch = img.crop(box)
        
        if self.colorJitter:
            patch = self.colorJitter(patch)
        
        rot_deg = random.uniform(*self.rotation)
        patch = patch.convert('RGBA').rotate(rot_deg, expand=True)
        
        to_location_h = int(random.uniform(0, h - patch.size[0]))
        to_location_w = int(random.uniform(0, w - patch.size[1]))
        
        mask = patch.split()[-1]
        patch = patch.convert('RGB')
        
        augmented = img.copy()
        augmented.paste(patch, (to_location_w, to_location_h), mask=mask)
        
        return super().__call__(img, augmented)
    
    
class CutPasteUnion:
    def __init__(self, **kwargs):
        self.normal = CutPasteNormal(**kwargs)
        self.scar = CutPasteScar(**kwargs)
    
    def __call__(self, img):
        r = random.uniform(0, 1)
        
        if r < 0.5:
            return self.normal(img)
        
        else:
            return self.scar(img)
        
class CutPaste3Way:
    def __init__(self, **kwargs):
        self.normal = CutPasteNormal(**kwargs)
        self.scar = CutPasteScar(**kwargs)
        
    def __call__(self, img):
        org, cutpaste_normal = self.normal(img)
        _, cutpaste_scar = self.scar(img)
        
        return org, cutpaste_normal, cutpaste_scar 
