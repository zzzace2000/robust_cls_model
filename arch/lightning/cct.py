from os.path import exists as pexists
from os.path import join as pjoin
import os

import numpy as np
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from .base import EpochBaseLightningModel
from .. import models
from ..data.cct_datasets import MyCCT_Dataset


class CCTLightningModel(EpochBaseLightningModel):
    def init_setup(self):
        if 'data_ratio' not in self.hparams:
            self.hparams.data_ratio = 1

        # Resnet 50
        head_size = 30 if self.hparams.cf.startswith('channels') else 15
        self.model = models.KNOWN_MODELS['BiT-S-R50x1'](
            head_size=head_size,
            zero_head=True)
        self.my_logger.info("Fine-tuning from BiT")
        if not pexists(f"models/BiT-S-R50x1.npz"):
            os.system(f'wget -O models/BiT-S-R50x1.npz '
                      f'https://storage.googleapis.com/bit_models/BiT-S-R50x1.npz')

        self.model.load_from(np.load(f"models/BiT-S-R50x1.npz"))

        ## Resnet50 has reused the ReLU and not able to derive DeepLift
        # self.model = resnet50(pretrained=True)
        # self.model.fc = torch.nn.Linear(2048, 15, bias=True)
        # torch.nn.init.zeros_(self.model.fc.weight)
        # torch.nn.init.zeros_(self.model.fc.bias)

    def get_inpainting_model(self, inpaint):
        if inpaint == 'cagan':
            return None
        return super().get_inpainting_model(inpaint)

    def configure_optimizers(self):
        optim = torch.optim.RMSprop(
            self.model.parameters(), lr=self.hparams.base_lr,
            momentum=0.9)
        scheduler = {
            # Total 50 epochs
            'scheduler': torch.optim.lr_scheduler.MultiStepLR(
                optim, milestones=[15, 30, 45], gamma=0.1),
            'interval': 'epoch',
        }
        return [optim], [scheduler]

    # @classmethod
    # def counterfactual_ce_loss(cls, logit, y, reduction='none'):
    #     '''
    #     If it's counterfactual, assign it to empty class 11
    #     '''
    #     assert (y < 0).all(), str(y)
    #     cf_y = 11 * torch.ones_like(y).long()
    #     return F.cross_entropy(logit, cf_y, reduction=reduction)

    def _make_train_val_dataset(self):
        cf_inpaint_dir = None
        if self.hparams.inpaint == 'cagan':
            cf_inpaint_dir = './datasets/cct/cagan/'

        train_d = MyCCT_Dataset(
            './datasets/cct/eccv_18_annotation_files/train_annotations.json',
            cf_inpaint_dir=cf_inpaint_dir,
            transform=MyCCT_Dataset.get_train_bbox_transform()
        )

        dr = self.hparams.data_ratio
        assert 0. < dr <= 1., 'Data ratio is invalid: ' + str(dr)
        if dr < 1.:
            train_d, _ = self.sub_dataset(train_d, num_subset=int(len(train_d) * dr))

        val_d = MyCCT_Dataset(
            './datasets/cct/eccv_18_annotation_files/cis_val_annotations.json',
            transform=MyCCT_Dataset.get_val_bbox_transform()
        )
        cis_test_d = MyCCT_Dataset(
            './datasets/cct/eccv_18_annotation_files/cis_test_annotations.json',
            transform=MyCCT_Dataset.get_val_bbox_transform()
        )
        trans_test_d = MyCCT_Dataset(
            './datasets/cct/eccv_18_annotation_files/trans_test_annotations.json',
            transform=MyCCT_Dataset.get_val_bbox_transform()
        )

        val_sets_names = ['val', 'cis_test', 'trans_test']
        return train_d, [val_d, cis_test_d, trans_test_d], val_sets_names

    def is_data_ratio_exp(self):
        return self.hparams.data_ratio != 1. or '_dr' in self.hparams.name

    @classmethod
    def add_model_specific_args(cls, parser):
        parser.add_argument("--max_epochs", type=int, default=50)
        parser.add_argument("--batch", type=int, default=64,
                            help="Batch size.")
        parser.add_argument("--val_batch", type=int, default=256,
                            help="Batch size.")
        parser.add_argument("--batch_split", type=int, default=1,
                            help="Number of batches to compute gradient on before updating weights.")
        parser.add_argument("--base_lr", type=float, default=0.003)
        parser.add_argument("--pl_model", type=str, default=cls.__name__)
        parser.add_argument("--reg_anneal", type=float, default=0.)

        parser.add_argument("--data_ratio", type=float, default=1.,
                            help='Specifies how many data to use. '
                                 'Default is 1: it means just using the cct dataset.'
                                 'It can be only use 0~1.')
        return parser

    def pl_trainer_args(self):
        checkpoint_callback = ModelCheckpoint(
            filepath=pjoin(self.hparams.logdir, self.hparams.name, '{epoch}'),
            save_top_k=1,
            save_last=True,
            verbose=True,
            mode='max',
            monitor='val_auc',
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
