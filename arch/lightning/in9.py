import os
from os.path import exists as pexists
from os.path import join as pjoin

import numpy as np
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from .base import EpochBaseLightningModel
from .. import models
from ..data.imagenet_datasets import MyConcatDataset
from ..data.imagenet_datasets import MyImageFolder, MyImagenetBoundingBoxFolder
from ..data.in9_datasets import IN9Dataset
from ..inpainting.Baseline import TileInpainter


class IN9LightningModel(EpochBaseLightningModel):
    train_dataset = 'original'
    milestones = [6, 12, 18]
    max_epochs = 25

    def init_setup(self):
        # Resnet 50
        if 'arch' not in self.hparams:
            self.hparams.arch = 'BiT-S-R50x1'
        if 'data_ratio' not in self.hparams:
            self.hparams.data_ratio = 1.
        if 'bbox_noise' not in self.hparams:
            self.hparams.bbox_noise = 0.

        head_size = 18 if self.hparams.cf.startswith('channels') else 9
        self.model = models.KNOWN_MODELS[self.hparams.arch](
            head_size=head_size,
            zero_head=False)
        if self.hparams.finetune:
            if not pexists(f"models/{self.hparams.arch}.npz"):
                os.system(f'wget -O models/{self.hparams.arch}.npz '
                          f'https://storage.googleapis.com/bit_models/{self.hparams.arch}.npz')

            self.my_logger.info("Fine-tuning from BiT")
            self.model.load_from(np.load(f"models/{self.hparams.arch}.npz"))

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.model.parameters(), lr=self.hparams.base_lr,
            momentum=0.9, weight_decay=1e-4)
        scheduler = {
            # Total 50 epochs
            'scheduler': torch.optim.lr_scheduler.MultiStepLR(
                optim, milestones=self.milestones, gamma=0.1),
            'interval': 'epoch',
        }
        return [optim], [scheduler]

    def get_inpainting_model(self, inpaint):
        if inpaint == 'cagan':
            return None
        if inpaint == 'tile' and self.hparams.mask == 'seg':
            return TileInpainter(use_bbox_to_mask=True)
        return super().get_inpainting_model(inpaint)

    def _make_train_val_dataset(self):
        cf_inpaint_dir = None
        if self.hparams.inpaint == 'cagan':
            if self.hparams.bbox_noise == 0.:
                cf_inpaint_dir = './datasets/bg_challenge/train/original_bbox_cf_cagan/train/'
            else:
                cf_inpaint_dir = './datasets/bg_challenge/train/cf_cagan_bbox_noise_%s/train/' \
                                 % self.hparams.bbox_noise

        if self.hparams.mask == 'bbox':
            assert self.hparams.bbox_noise == 0.
            train_d = MyImagenetBoundingBoxFolder(
                './datasets/bg_challenge/train/%s/train/' % self.train_dataset,
                './datasets/imagenet/LOC_train_solution.csv',
                cf_inpaint_dir=cf_inpaint_dir,
                transform=MyImagenetBoundingBoxFolder.get_train_transform(
                    self.hparams.test_run))
        else:
            train_d = IN9Dataset(
                './datasets/bg_challenge/train/%s/train/' % self.train_dataset,
                no_fg_dir='./datasets/bg_challenge/train/no_fg/train/',
                cf_inpaint_dir=cf_inpaint_dir,
                bbox_noise=self.hparams.bbox_noise,
                transform=IN9Dataset.get_train_transform(self.hparams.test_run)
            )

        if self.hparams.data_ratio == 1.:
            pass
        elif 0. < self.hparams.data_ratio < 1.:
            num_data = int(len(train_d) * self.hparams.data_ratio)
            train_d, _ = self.sub_dataset(train_d, num_data)
        elif (self.hparams.data_ratio > 1. or self.hparams.data_ratio == -1) \
                and self.train_dataset == 'original':
            orig_filenames = set()
            with open('./datasets/bg_challenge/train/original/train_filenames') as fp:
                for line in fp:
                    orig_filenames.add(line.strip())

            def is_valid_file(path):
                return os.path.basename(path) not in orig_filenames

            more_train_d = MyImageFolder(
                './datasets/bg_challenge/train/in9l/train/',
                is_valid_file=is_valid_file,
                transform=MyImageFolder.get_train_transform(self.hparams.test_run))

            if self.hparams.data_ratio > 1.:
                more_data = self.hparams.data_ratio - 1.
                num_data = int(len(train_d) * more_data)
                if num_data < len(more_train_d):
                    more_train_d, _ = self.sub_dataset(more_train_d, num_data)

            train_d = MyConcatDataset([train_d, more_train_d])
        else:
            if self.hparams.data_ratio != 1.:
                raise NotImplementedError(
                    'Data ratio is wronly specified: ' + str(self.hparams.data_ratio))

        val_d = MyImageFolder(
            './datasets/bg_challenge/train/%s/val/' % self.train_dataset,
            transform=MyImageFolder.get_val_transform(self.hparams.test_run))
        orig_test_d = MyImageFolder(
            './datasets/bg_challenge/test/original/val/',
            transform=MyImageFolder.get_val_transform(self.hparams.test_run))
        mixed_same_test_d = MyImageFolder(
            './datasets/bg_challenge/test/mixed_same/val/',
            transform=MyImageFolder.get_val_transform(self.hparams.test_run))
        mixed_rand_test_d = MyImageFolder(
            './datasets/bg_challenge/test/mixed_rand/val/',
            transform=MyImageFolder.get_val_transform(self.hparams.test_run))
        mixed_next_test_d = MyImageFolder(
            './datasets/bg_challenge/test/mixed_next/val/',
            transform=MyImageFolder.get_val_transform(self.hparams.test_run))

        val_ds = [
            val_d, orig_test_d, mixed_same_test_d,
            mixed_rand_test_d, mixed_next_test_d]
        val_sets_names = [
            'val', 'orig', 'mixed_same', 'mixed_rand', 'mixed_next'
        ]
        return train_d, val_ds, val_sets_names

    def is_data_ratio_exp(self):
        return self.hparams.data_ratio != 1. or '_dr' in self.hparams.name

    def is_bbox_noise_exp(self):
        return self.hparams.bbox_noise > 0. or '_bn' in self.hparams.name

    @classmethod
    def add_model_specific_args(cls, parser):
        # To use bbox as mask or segmentation mask
        parser.add_argument("--mask", type=str, default='seg',
                            choices=['bbox', 'seg'])
        parser.add_argument("--arch", type=str, default='BiT-S-R50x1')
        parser.add_argument("--max_epochs", type=int, default=cls.max_epochs)
        parser.add_argument("--batch", type=int, default=32,
                            help="Batch size.")
        parser.add_argument("--val_batch", type=int, default=256,
                            help="Batch size.")
        parser.add_argument("--batch_split", type=int, default=1,
                            help="Number of batches to compute gradient on before updating weights.")
        parser.add_argument("--base_lr", type=float, default=0.05)
        parser.add_argument("--pl_model", type=str, default=cls.__name__)
        parser.add_argument("--reg_anneal", type=float, default=0.)

        parser.add_argument("--data_ratio", type=float, default=1.,
                            help='Specifies how many data to use. '
                                 'Default is 1: it means just using the original dataset.'
                                 'If bigger than 1, e.g. 2, then it adds 1x more data from'
                                 'in9l dataset. If it is -1, then it uses all data in9l.')
        parser.add_argument("--bbox_noise", type=float, default=0.,
                            help='If bigger than 0, we randomly shuffle the foreground mask')
        return parser

    def pl_trainer_args(self):
        checkpoint_callback = ModelCheckpoint(
            filepath=pjoin(self.hparams.logdir, self.hparams.name, '{epoch}'),
            save_top_k=1,
            save_last=True,
            verbose=True,
            mode='max',
            monitor='val_acc1',
        )

        args = dict()
        args['max_epochs'] = self.hparams.max_epochs
        args['checkpoint_callback'] = checkpoint_callback
        last_ckpt = pjoin(self.hparams.logdir, self.hparams.name, 'last.ckpt')
        if pexists(last_ckpt):
            args['resume_from_checkpoint'] = last_ckpt
        return args

    def get_grad_cam_layer(self):
        return self.model.head[1]


class IN9LLightningModel(IN9LightningModel):
    train_dataset = 'in9l'
    milestones = [5, 10, 15]
    max_epochs = 20
