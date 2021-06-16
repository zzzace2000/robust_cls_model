from torchvision.datasets import ImageFolder
import os
from os.path import join as pjoin, exists as pexists
import numpy as np
import torch
import random
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.dataset import Dataset, ConcatDataset, Subset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import Sampler
import bisect
from torch.nn.utils.rnn import pad_sequence
from . import bbox_utils
import torchvision as tv


class ImagenetBoundingBoxFolder(ImageFolder):
    ''' Custom loader that loads images with bounding box '''

    def __init__(self, root, bbox_file, is_valid_file=None,
                 cf_inpaint_dir=None,
                 only_bbox_imgs=False,
                 **kwargs):
        ''' bbox_file points to either `LOC_train_solution.csv` or `LOC_val_solution.csv` '''
        self.cf_inpaint_dir = cf_inpaint_dir
        self.coord_dict = self.parse_coord_dict(bbox_file)
        if is_valid_file is None and only_bbox_imgs is True:
            is_valid_file = lambda path: os.path.basename(path) in self.coord_dict
        super().__init__(root, is_valid_file=is_valid_file, **kwargs)

    @staticmethod
    def parse_coord_dict(data_file):
        # map from ILSVRC2012_val_00037956 to ('n03995372', [85 1 499 272])
        coord_dict = {}
        with open(data_file) as fp:
            fp.readline()
            for line in fp:
                line = line.strip().split(',')
                filename = '%s.JPEG' % line[0]
                tmp = line[1].split(' ')

                xs, ys, ws, hs = [], [], [], []
                the_first_class = tmp[0]
                for i in range(len(tmp) // 5):
                    the_class = tmp[i * 5]
                    if the_class != the_first_class:
                        continue

                    # The string is: n0133595 x1 y1 x2 y2
                    [x1, y1, x2, y2] = tmp[(i * 5 + 1):(i * 5 + 5)]

                    # parse it in x, y, w, h
                    xs.append(int(x1))
                    ys.append(int(y1))
                    ws.append((int(x2) - int(x1)))
                    hs.append((int(y2) - int(y1)))

                # Only take the first bounding box which is the ground truth
                coord_dict[filename] = dict(
                    xs=torch.LongTensor(xs),
                    ys=torch.LongTensor(ys),
                    ws=torch.LongTensor(ws),
                    hs=torch.LongTensor(hs),
                )

        return coord_dict

    def __getitem__(self, index):
        """
        Override this to return the bounding box as well
        """
        path, target = self.samples[index]
        imgs = self.loader(path)

        # Append the bounding box in the 4th channel
        filename = os.path.basename(path)
        if filename not in self.coord_dict:
            sample = dict(
                imgs=imgs,
                xs=torch.tensor([-1]),
                ys=torch.tensor([-1]),
                ws=torch.tensor([-1]),
                hs=torch.tensor([-1]),
            )
        else:
            bbox = self.coord_dict[filename]
            sample = dict(imgs=imgs,
                xs=bbox['xs'].clone(),
                ys=bbox['ys'].clone(),
                ws=bbox['ws'].clone(),
                hs=bbox['hs'].clone(),
            )

        if self.cf_inpaint_dir is not None:
            cls_name = self.classes[target]
            img_name = path.split("/")[-1]
            cf_path = pjoin(self.cf_inpaint_dir, cls_name, img_name)
            sample['imgs_cf'] = self.loader(cf_path) if pexists(cf_path) else None

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

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


class MyHackSampleSizeMixin(object):
    '''
    Custom dataset to hack the lightning framework.
    To train a fixed number of steps, since lightning only supports
    epoch-based training. So this hack is to return a dataset that
    has special length to make resulting loader run for an epoch.
    '''
    def __init__(self, *args, my_num_samples=None, **kwargs):
        self.my_num_samples = my_num_samples
        super().__init__(*args, **kwargs)

    def __len__(self):
        if self.my_num_samples is None:
            return super().__len__()

        return self.my_num_samples

    def __getitem__(self, index):
        actual_len = super().__len__()
        get_item_func = super().__getitem__
        if isinstance(index, list):
            return [get_item_func(i % actual_len) for i in index]
        return get_item_func(index % actual_len)


class MyImageFolder(MyHackSampleSizeMixin, ImageFolder):
    def make_loader(self, batch_size, shuffle, workers, pin_memory=True, **kwargs):
        return DataLoader(
            self, batch_size=batch_size, shuffle=shuffle,
            num_workers=workers, pin_memory=pin_memory, **kwargs)

    @property
    def is_bbox_folder(self):
        return False

    @classmethod
    def get_train_transform(self, test_run=False):
        precut, cut = 256, 224
        if test_run:
            precut, cut = 16, 14

        train_bbox_tx = tv.transforms.Compose([
            tv.transforms.Resize((precut, precut)),
            tv.transforms.RandomCrop((cut, cut)),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ColorJitter(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        return train_bbox_tx

    @classmethod
    def get_val_transform(cls, test_run=False):
        cut = 14 if test_run else 224

        val_bbox_tx = tv.transforms.Compose([
            tv.transforms.Resize((cut, cut)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        return val_bbox_tx


class MyImagenetBoundingBoxFolder(MyHackSampleSizeMixin, ImagenetBoundingBoxFolder):
    def make_loader(self, batch_size, shuffle, workers, pin_memory=True, **kwargs):
        # if self.my_num_samples is not None:
        #     self.my_num_samples //= 2
        # batch_size = batch_size // 2
        return DataLoader(
            self, batch_size=batch_size, shuffle=shuffle,
            num_workers=workers, pin_memory=pin_memory,
            collate_fn=bbox_utils.bbox_collate, **kwargs)

    @property
    def is_bbox_folder(self):
        return True


class MySubset(MyHackSampleSizeMixin, Subset):
    def __init__(self, dataset, indices, transform=None):
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def make_loader(self, batch_size, shuffle, workers, **kwargs):
        the_dataset = self.dataset
        while isinstance(the_dataset, MySubset):
            the_dataset = the_dataset.dataset

        return the_dataset.__class__.make_loader(
            self, batch_size, shuffle, workers, **kwargs)

    @property
    def is_bbox_folder(self):
        return self.dataset.is_bbox_folder

    @property
    def classes(self):
        return self.dataset.classes


class MyConcatDataset(MyHackSampleSizeMixin, ConcatDataset):
    def make_loader(self, batch_size, shuffle, workers, pin_memory=True, **kwargs):
        '''
        Possibly 1 bbox folder and 1 img folder, or 2 img folders
        '''
        my_self = self
        while isinstance(my_self, MySubset):
            my_self = my_self.dataset
        # all bbox folder or not-bbox folder
        if np.all(self.is_bbox_folder) or not np.any(self.is_bbox_folder):
            return my_self.datasets[0].__class__.make_loader(
                self, batch_size, shuffle, workers,
                pin_memory=pin_memory,
                **kwargs)

        # 1 img folder and 1 bbox folder
        sampler = MyConcatDatasetSampler(
            self, batch_size,
            my_num_samples=self.my_num_samples,
            shuffle=shuffle)
        return DataLoader(
            self, batch_size=None, sampler=sampler,
            num_workers=workers, pin_memory=pin_memory,
            collate_fn=bbox_utils.bbox_collate, **kwargs)

    @property
    def use_my_batch_sampler(self):
        return (not np.all(self.is_bbox_folder) and np.any(self.is_bbox_folder))

    @property
    def is_bbox_folder(self):
        ''' return a list of bbox folder for its underlying datasets '''
        return [d.is_bbox_folder for d in self.datasets]

    @property
    def classes(self):
        return self.datasets[0].classes


class MyConcatDatasetSampler(Sampler):
    '''
    For each sub dataset, it loops through each dataset randomly with
    batch size, but does not mix different dataset within a same batch
    '''
    def __init__(self, data_source, batch_size, my_num_samples=None, shuffle=True):
        assert isinstance(data_source, ConcatDataset), \
            'Wrong data source with type ' + type(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.my_num_samples = my_num_samples
        self.batch_len = int(sum([
            len(d) // batch_size
            for d in self.data_source.datasets]))
        self.gen_func = torch.randperm if shuffle else torch.arange

    def __iter__(self):
        cs = self.data_source.cumulative_sizes

        def idxes_generator(s, e):
            bs = self.batch_size
            idxes = self.gen_func(e - s) + s

            for s in range(0, len(idxes), bs):
                yield idxes[s:(s + bs)].tolist()

        while True:
            # randomly sample batch from each dataset
            generators = [
                iter(idxes_generator(s, e))
                for s, e in zip([0] + cs[:-1], cs)
            ]
            while len(generators) > 0:
                g_idx = random.randint(0, len(generators) - 1)
                try:
                    yield next(generators[g_idx])
                except StopIteration:
                    generators.pop(g_idx)
                    continue

    def __len__(self):
        if self.my_num_samples is not None:
            return self.my_num_samples
        return self.batch_len


class MyFactualAndCFDatasetBase(Dataset):
    def __init__(self, factual_folder, cf_folder):
        assert len(factual_folder) == len(cf_folder)
        assert isinstance(factual_folder, MyImagenetBoundingBoxFolder)
        assert isinstance(cf_folder, MyImageFolder)

        self.factual_folder = factual_folder
        self.cf_folder = cf_folder

    def __getitem__(self, item):
        sample, y = self.factual_folder[item]
        x_cf, _ = self.cf_folder[item]
        sample['imgs_cf'] = x_cf

        return sample, y

    def __len__(self):
        return len(self.factual_folder)

    def make_loader(self, batch_size, shuffle, workers, pin_memory=True, **kwargs):
        # if self.my_num_samples is not None:
        #     self.my_num_samples //= 2
        # batch_size = batch_size // 2
        return DataLoader(
            self, batch_size=batch_size, shuffle=shuffle,
            num_workers=workers, pin_memory=pin_memory,
            collate_fn=bbox_utils.bbox_collate, **kwargs)

    @property
    def classes(self):
        return self.factual_folder.classes


class MyFactualAndCFDataset(MyHackSampleSizeMixin, MyFactualAndCFDatasetBase):
    pass


class MyImageNetODataset(MyConcatDataset):
    '''
    Generate an Imagenet-o dataset.
    Idea is to combine two imagefolder datasets from the 2 directory.
    Then set the images in the imagenet-o with target 1, and the val
    imagenet images (w/ 200 classes) with target 0.
    '''

    def __init__(self, imageneto_dir, val_imgnet_dir, transform):
        imageneto = MyImageFolder(imageneto_dir, transform=transform)
        val_imgnet = MyImageFolder(val_imgnet_dir, transform=transform)

        super().__init__([val_imgnet, imageneto])

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        x, _ = self.datasets[dataset_idx][sample_idx]
        y = dataset_idx # 0 means normal, 1 means outlier (imgnet-o)
        return x, y

