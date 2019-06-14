import torch
import pandas as pd
from torchvision import transforms
import os
from PIL import Image


BASIC_IMAGE_T = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

TRAIN_IMAGE_T = transforms.Compose([
    transforms.RandomAffine(degrees=360, shear=11),
    transforms.RandomResizedCrop(128, scale=(.9, 1.1)),
    transforms.ColorJitter(brightness=0.11, contrast=0.11, saturation=0.11, hue=0.11),
    transforms.RandomGrayscale(.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class WhaleDataset(torch.utils.data.Dataset):
    def __init__(self, path, image_transformation=TRAIN_IMAGE_T):
        self.path = path
        df = pd.read_csv(os.path.join(path, 'train.csv'), index_col=False)
        n_samples = len(df)
        df = pd.concat([df] * 2, ignore_index=True)
        df['flipped'] = [False] * n_samples + [True] * n_samples
        df.loc[n_samples:, 'Id'] = df['Id'][n_samples:].apply(lambda x: 'flipped_'+x)
        df['idx'] = range(len(df))
        self._all_records = df
        grouping = self._all_records.groupby(['Id'])
        self._by_class = grouping.groups
        self.index_to_class = list(self._by_class.keys())
        self.index_to_class.sort()
        self.class_to_index = {k:i for i,k in enumerate(self.index_to_class)}
        anchor = grouping['Image'].count() > 9
        anchor['new_whale'] = False
        anchor['flipped_new_whale'] = False
        anchor_index = anchor[df.Id]
        self._records = df[list(anchor_index)]
        self._negative_records = df[list(~anchor_index)]
        self.image_t = image_transformation
        self._label_encoder = self.class_to_index
    
    def __len__(self):
        return len(self._records)
    
    def __getitem__(self, idr):
        if isinstance(idr, torch.Tensor):
            idr = idr.item()
        r = self._records.iloc[idr]
        idx = r['idx']
        op = r['Image']
        op_id = r['Id']
        flipped = r['flipped']
        sample = [
            self.process_image(op),
            torch.tensor(self.class_to_index[op_id]),
        ]
        if flipped:
            sample[0] = self.flip_image(sample[0])
        return tuple(sample)
    
    def flip_image(self, x):
        return torch.flip(x, [1, 2])
    
    def process_image(self, image_id):
        im = Image.open(os.path.join(self.path, 'train', image_id)).convert("RGB")
        return self.image_t(im)

