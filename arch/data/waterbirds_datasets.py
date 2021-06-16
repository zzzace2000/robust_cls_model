from os.path import join as pjoin

import pandas as pd
import torchvision as tv
from PIL import Image
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import torch
import numpy as np

from . import bbox_utils


class WaterbirdDataset(Dataset):
    ''' Custom loader that loads images with bounding box '''

    def __init__(self,
                 mode,
                 type='all',
                 train_dir='./datasets/waterbird/waterbird_complete95_forest2water2/',
                 seg_dir='./datasets/waterbird/segmentations/',
                 only_images=False,
                 cf_inpaint_dir=None,
                 transform=None):
        '''
        Data loader for waterbird dataset
        :param mode: choose from ['train', 'val', 'test']
        :param type: choose from ['same', 'flip', 'all']. Same means to choose
            images with the same backgrounds as labels. 'Flip' means
            to choose images with opposite backgrounds than labels.
        :param train_dir:
        :param seg_dir:
        :param cf_inpaint_dir:
        :param transform:
        '''
        assert mode in ['train', 'val', 'test', 'all']
        assert type in ['same', 'flip', 'all']
        super().__init__()
        self.mode = mode
        self.train_dir = train_dir
        self.seg_dir = seg_dir
        self.only_images = only_images
        self.cf_inpaint_dir = cf_inpaint_dir
        self.transform = transform

        self.metadata = pd.read_csv(pjoin(train_dir, 'metadata.csv'))
        if mode != 'all':
            mode_idx = {'train': 0, 'val': 1, 'test': 2}[mode]
            self.metadata = self.metadata[self.metadata.split == mode_idx]

        if type == 'same':
            is_kept = (self.metadata.y == self.metadata.place)
            self.metadata = self.metadata[is_kept]
        elif type == 'flip':
            is_kept = (self.metadata.y != self.metadata.place)
            self.metadata = self.metadata[is_kept]

    def __getitem__(self, index):
        """
        Override this to return the bounding box as well
        """
        f_name = self.metadata.img_filename.iloc[index]
        img_path = pjoin(self.train_dir, f_name)
        img = Image.open(img_path).convert('RGB')
        target = self.metadata.y.iloc[index]

        if self.only_images:
            img = self.transform(img)
            return img, target

        seg_path = pjoin(self.seg_dir, f_name.replace('.jpg', '.png'))
        seg = Image.open(seg_path).convert('RGB')

        sample = dict()
        sample['imgs'] = img
        sample['masks'] = seg

        # extract x, y, w, h
        seg_np = np.asarray(seg) / 255
        tmp = np.arange(seg_np.shape[0])[seg_np.all(axis=2).any(axis=1)]
        sample['ys'], sample['hs'] = torch.tensor(tmp[0]), \
                                     torch.tensor(tmp[-1] - tmp[0] + 1)
        tmp = np.arange(seg_np.shape[1])[seg_np.all(axis=2).any(axis=0)]
        sample['xs'], sample['ws'] = torch.tensor(tmp[0]), \
                                     torch.tensor(tmp[-1] - tmp[0] + 1)

        if self.cf_inpaint_dir is not None:
            cf_path = pjoin(self.cf_inpaint_dir, f_name)
            sample['imgs_cf'] = Image.open(cf_path).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return self.metadata.shape[0]

    @classmethod
    def get_train_transform(self, test_run=False):
        precut, cut = 256, 224
        if test_run:
            precut, cut = 16, 14

        train_bbox_tx = tv.transforms.Compose([
            bbox_utils.Resize((precut, precut)),
            bbox_utils.RandomCrop((cut, cut)),
            bbox_utils.RandomHorizontalFlip(),
            bbox_utils.ColorJitter(),
            bbox_utils.ToTensor(),
            bbox_utils.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        return train_bbox_tx

    @classmethod
    def get_val_transform(self, test_run=False):
        cut = 14 if test_run else 224

        val_bbox_tx = tv.transforms.Compose([
            bbox_utils.Resize((cut, cut)),
            bbox_utils.ToTensor(),
            bbox_utils.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        return val_bbox_tx

    def make_loader(self, batch_size, shuffle, workers, pin_memory=True, **kwargs):
        collate_fn = None
        if not self.only_images:
            collate_fn = bbox_utils.bbox_collate

        return DataLoader(
            self, batch_size=batch_size, shuffle=shuffle,
            num_workers=workers, pin_memory=pin_memory,
            collate_fn=collate_fn, **kwargs)

    @property
    def is_bbox_folder(self):
        return True
