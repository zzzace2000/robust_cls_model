from os.path import exists as pexists
from os.path import join as pjoin
import os

import numpy as np
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from .base import EpochBaseLightningModel
from .. import models
from ..data.imagenet_datasets import MyImageFolder
from ..data.waterbirds_datasets import WaterbirdDataset


class WaterbirdLightningModel(EpochBaseLightningModel):
    max_epochs = 40

    def init_setup(self):
        # Resnet 50
        if 'arch' not in self.hparams:
            self.hparams.arch = 'BiT-S-R50x1'
        if 'data_ratio' not in self.hparams:
            self.hparams.data_ratio = 1.

        self.model = models.KNOWN_MODELS[self.hparams.arch](
            head_size=2,
            zero_head=False)
        if self.hparams.finetune:
            self.my_logger.info("Fine-tuning from BiT")
            if not pexists(f"models/BiT-S-R50x1.npz"):
                os.system(f'wget -O models/BiT-S-R50x1.npz '
                          f'https://storage.googleapis.com/bit_models/BiT-S-R50x1.npz')

            self.model.load_from(np.load(f"models/BiT-S-R50x1.npz"))

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.model.parameters(), lr=self.hparams.base_lr,
            momentum=0.9, weight_decay=1e-4)

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim,
                'min',
                factor=0.1,
                patience=1,
                threshold=0.0001,
                min_lr=0,
                eps=1e-08),
            'interval': 'epoch',
        }
        return [optim], [scheduler]

    def get_inpainting_model(self, inpaint):
        if inpaint == 'cagan':
            return None
        return super().get_inpainting_model(inpaint)

    def _make_train_val_dataset(self):
        cf_inpaint_dir = None
        if self.hparams.inpaint == 'cagan':
            cf_inpaint_dir = '../datasets/waterbird/cagan/'

        train_d = WaterbirdDataset(
            mode='train',
            cf_inpaint_dir=cf_inpaint_dir,
            transform=WaterbirdDataset.get_train_transform(
                self.hparams.test_run)
        )

        if self.hparams.data_ratio == 1.:
            pass
        elif 0. < self.hparams.data_ratio < 1.:
            num_data = int(len(train_d) * self.hparams.data_ratio)
            train_d, _ = self.sub_dataset(train_d, num_data)
        else:
            raise NotImplementedError(
                'Data ratio is wronly specified: '
                + str(self.hparams.data_ratio))

        val_tx = MyImageFolder.get_val_transform(self.hparams.test_run)
        val_d = WaterbirdDataset(
            mode='val',
            only_images=True,
            transform=val_tx)
        orig_test_d = WaterbirdDataset(
            mode='test',
            type='same',
            only_images=True,
            transform=val_tx)
        flip_test_d = WaterbirdDataset(
            mode='test',
            type='flip',
            only_images=True,
            transform=val_tx)

        val_ds = [
            val_d, orig_test_d, flip_test_d]
        val_sets_names = [
            'val', 'orig', 'flip',
        ]
        return train_d, val_ds, val_sets_names

    def is_data_ratio_exp(self):
        return self.hparams.data_ratio != 1. or '_dr' in self.hparams.name

    @classmethod
    def add_model_specific_args(cls, parser):
        parser.add_argument("--arch", type=str, default='BiT-S-R50x1')
        parser.add_argument("--max_epochs", type=int, default=cls.max_epochs)
        parser.add_argument("--batch", type=int, default=32,
                            help="Batch size.")
        parser.add_argument("--val_batch", type=int, default=256,
                            help="Batch size.")
        parser.add_argument("--batch_split", type=int, default=1,
                            help="Number of batches to compute gradient on before updating weights.")
        parser.add_argument("--base_lr", type=float, default=8e-4)
        parser.add_argument("--pl_model", type=str, default=cls.__name__)
        parser.add_argument("--reg_anneal", type=float, default=0.)

        parser.add_argument("--data_ratio", type=float, default=1.,
                            help='Specifies how many data to use. '
                                 'Default is 1: it means just using the original dataset.'
                                 'If bigger than 1, e.g. 2, then it adds 1x more data from'
                                 'in9l dataset. If it is -1, then it uses all data in9l.')
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
