import torch
import pandas as pd
import torchvision
import kornia
from torchvision import transforms
import os
import random
from PIL import Image, ImageOps
import albumentations
from albumentations.pytorch import ToTensor
import numpy as np


FINALIZE_T = albumentations.Compose([
    albumentations.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ToTensor(),
])

BASIC_IMAGE_T = albumentations.Compose([
    albumentations.Resize(128, 128),
    FINALIZE_T
])

AUG_IMAGE_T = albumentations.Compose([
    albumentations.GaussNoise(),
    albumentations.OneOf([
        albumentations.MotionBlur(),
        albumentations.MedianBlur(blur_limit=3),
        albumentations.Blur(blur_limit=3),
        albumentations.JpegCompression(quality_lower=90),
    ]),
    albumentations.OneOf([
        albumentations.CLAHE(clip_limit=2),
        albumentations.IAASharpen(),
        albumentations.RandomBrightnessContrast(),
    ]),
    albumentations.HueSaturationValue(p=0.3),
])


TRAIN_IMAGE_T = albumentations.Compose([
    albumentations.Resize(128, 128),
    AUG_IMAGE_T,
    FINALIZE_T
])

    
class WhaleDataset(torch.utils.data.Dataset):
    def __init__(self, path, image_transformation=TRAIN_IMAGE_T, include_flips=True, min_instance_count=1):
        self.path = path
        df = pd.read_csv(os.path.join(path, 'train.csv'), index_col=False)
        if include_flips:
            n_samples = len(df)
            df = pd.concat([df] * 2, ignore_index=True)
            df['flipped'] = [False] * n_samples + [True] * n_samples
            df.loc[n_samples:, 'Id'] = df['Id'][n_samples:].apply(lambda x: 'flipped_'+x)
        else:
            df['flipped'] = False
        df['idx'] = range(len(df))
        self._all_records = df
        grouping = self._all_records.groupby(['Id'])
        self._by_class = grouping.groups
        anchor = grouping['Image'].count() >= min_instance_count
        anchor['new_whale'] = False
        if include_flips:
            anchor['flipped_new_whale'] = False
        anchor_index = anchor[df.Id]
        self.index_to_class = list(map(lambda i:i[0], filter(lambda i:i[1], anchor.items())))
        self.index_to_class.sort()
        self.class_to_index = {k:i for i,k in enumerate(self.index_to_class)}
        self._records = df[list(anchor_index)]
        self._negative_records = df[list(~anchor_index)]
        self.image_t = image_transformation

    def __len__(self):
        return len(self._records)

    def __getitem__(self, idr):
        if isinstance(idr, torch.Tensor):
            idr = idr.item()
        r = self._records.iloc[idr]
        return self.record_to_sample(r)
    
    def record_to_sample(self, r):
        idx = r['idx']
        op = r['Image']
        op_id = r['Id']
        flipped = r['flipped']
        return (
            self.process_image(op, flipped),
            torch.tensor(self.class_to_index[op_id]),
            idx,
        )

    def read_image(self, image_id, flip):
        img = Image.open(os.path.join(self.path, 'train', image_id)).convert("RGB")
        if flip:
            img = transforms.functional.hflip(img)
        return img

    def process_image(self, image_id, flip):
        im = np.array(self.read_image(image_id, flip))
        return self.image_t(image=im)['image']


class TripletWhaleDataset(WhaleDataset):
    def __getitem__(self, idr):
        if isinstance(idr, torch.Tensor):
            idr = idr.item()
        r = self._records.iloc[idr]
        pr = self.get_positive_record(r['idx'])
        nr = self.get_negative_record(r['idx'])
        return (
            *self.record_to_sample(r),
            *self.record_to_sample(pr),
            *self.record_to_sample(nr),
        )
    
    def get_positive_record(self, idx):
        k = self._all_records.iloc[idx]['Id']
        assert len(self._by_class[k]) > 1
        p_idx = idx
        while p_idx == idx:
            p_idx = random.choice(self._by_class[k])
        return self._all_records.iloc[p_idx]
    
    def get_negative_record(self, idx):
        k = self._all_records.iloc[idx]['Id']
        p_k = k
        while p_k == k:
            k_idx = random.randint(0, len(self.index_to_class)-1)
            p_k = self.index_to_class[k_idx]
        p_idx = random.choice(self._by_class[p_k])
        return self._all_records.iloc[p_idx]
        