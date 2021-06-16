'''
The loader for Caltech Camera Traps (CCT) datasets.
https://beerys.github.io/CaltechCameraTraps/
'''

import json
import os
from os.path import join as pjoin, exists as pexists

import numpy as np
import pandas as pd
import torch
import torchvision as tv
from torchvision.datasets.folder import default_loader

from . import bbox_utils

thispath = os.path.dirname(os.path.realpath(__file__))


class MyCCT_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 json_file='./datasets/cct/eccv_18_annotation_files/train_annotations.json',
                 cct_img_folder='./datasets/cct/eccv_18_all_images_sm/',
                 cf_inpaint_dir=None,
                 remove_empty_img=True,
                 only_bbox_imgs=False,
                 no_bbox=False,
                 transform=None):
        super().__init__()
        self.json_file = json_file
        self.cct_img_folder = cct_img_folder
        self.cf_inpaint_dir = cf_inpaint_dir
        self.remove_empty_img = remove_empty_img
        self.only_bbox_imgs = only_bbox_imgs
        self.no_bbox = no_bbox
        self.transform = transform

        # Load json
        self.setup_data()

    def setup_data(self):
        tmp = json.load(open(self.json_file))
        # It has 'image_id', 'category_id', 'bbox', 'id'
        annotations = pd.DataFrame(tmp['annotations'])
        images = pd.DataFrame(tmp['images'])[['id', 'height', 'width']]

        # Merge bboxes for the same image
        annotations = annotations.groupby(['image_id', 'category_id']).apply(
            lambda x: np.nan if np.isnan(x.bbox.iloc[0]).any() else x.bbox.values.tolist())
        annotations.name = 'bbox'
        annotations = annotations.reset_index()
        self.annotations = pd.merge(annotations, images,
                                    how='left', left_on='image_id', right_on='id')
        self.annotations = self.annotations[[
            'image_id', 'category_id', 'bbox', 'height', 'width']].reset_index(drop=True)

        if self.remove_empty_img:
            self.annotations = self.annotations[self.annotations.category_id != 30]
        if self.only_bbox_imgs:
            self.annotations = self.annotations[~pd.isnull(self.annotations.bbox)]

        # setup category_id to the target id
        cat_df = pd.DataFrame(tmp['categories'])
        if self.remove_empty_img:
            cat_df = cat_df[cat_df.id != 30]

        tmp2 = cat_df.sort_values('id').reset_index(drop=True)
        self.y_to_cat_id = tmp2['id']
        self.cat_id_to_y = pd.Series(self.y_to_cat_id.index,
                                     self.y_to_cat_id.values)
        self.category_name = tmp2['name']

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        record = self.annotations.iloc[idx]
        img_path = pjoin(self.cct_img_folder, record['image_id'] + '.jpg')
        img = default_loader(img_path)

        target = self.cat_id_to_y[record.category_id]
        if self.no_bbox:
            assert self.cf_inpaint_dir is None, "Avoid myself being stupid."
            if self.transform is not None:
                img = self.transform(img)

            assert img is not None and target is not None, str(idx)
            return img, target

        result = dict()
        result['imgs'] = img
        if np.isnan(record.bbox).any():
            result['xs'] = result['ys'] = result['ws'] = result['hs'] = \
                torch.tensor([-1])
        else:
            w, h = record.width, record.height
            new_w, new_h = img.width, img.height

            bbox_xs, bbox_ys, bbox_ws, bbox_hs = [], [], [], []
            for (bbox_x, bbox_y, bbox_w, bbox_h) in record.bbox:
                bbox_xs.append(int(bbox_x))
                bbox_ys.append(int(bbox_y))
                bbox_ws.append(int(bbox_w))
                bbox_hs.append(int(bbox_h))
            bbox_xs = torch.LongTensor(bbox_xs)
            bbox_ys = torch.LongTensor(bbox_ys)
            bbox_ws = torch.LongTensor(bbox_ws)
            bbox_hs = torch.LongTensor(bbox_hs)

            # Do transformation!!!
            if w != new_w or h != new_h:
                bbox_xs, bbox_ys, bbox_ws, bbox_hs = bbox_utils.Resize.resize_bbox(
                    w, h, new_w, new_h, bbox_xs, bbox_ys, bbox_ws, bbox_hs)

            result['xs'], result['ys'], result['ws'], result['hs'] \
                = bbox_xs, bbox_ys, bbox_ws, bbox_hs

        if self.cf_inpaint_dir is not None:
            cf_path = pjoin(self.cf_inpaint_dir, record['image_id'] + '.jpg')
            result['imgs_cf'] = default_loader(cf_path) \
                if (result['xs'] != -1).all() and pexists(cf_path) \
                else None

        if self.transform is not None:
            result = self.transform(result)
        return result, target

    @property
    def is_bbox_folder(self):
        return not self.no_bbox

    @classmethod
    def get_train_bbox_transform(cls, test_run=False):
        orig_size = 224 if not test_run else 14

        train_tx = tv.transforms.Compose([
            bbox_utils.Resize(orig_size),
            bbox_utils.RandomCrop((orig_size, orig_size)),
            bbox_utils.RandomHorizontalFlip(),
            bbox_utils.ColorJitter(),
            bbox_utils.ToTensor(),
            bbox_utils.Normalize((0.5,), (0.5,)),
        ])
        return train_tx

    @classmethod
    def get_val_bbox_transform(cls, test_run=False):
        orig_size = 224 if not test_run else 14
        val_tx = tv.transforms.Compose([
            bbox_utils.Resize((orig_size, orig_size)),
            bbox_utils.ToTensor(),
            bbox_utils.Normalize((0.5,), (0.5,)),
        ])
        return val_tx

    def make_loader(self, batch_size, shuffle, workers, pin_memory=True, **kwargs):
        return torch.utils.data.DataLoader(
            self, batch_size=batch_size, shuffle=shuffle,
            num_workers=workers, pin_memory=pin_memory,
            collate_fn=bbox_utils.bbox_collate, **kwargs)
