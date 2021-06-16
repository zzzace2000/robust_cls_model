import pytorch_lightning as pl
import os
from os.path import join as pjoin
from os.path import exists as pexists
from collections import OrderedDict
import pandas as pd
import numpy as np
from argparse import Namespace
import sys

from ..utils import output_csv
from .base_imagenet import ImageNetLightningModel


class CSVRecordingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics

        csv_dict = OrderedDict()
        csv_dict['epoch'] = pl_module.current_epoch
        csv_dict['global_step'] = pl_module.global_step
        # In the metrics csv, the epoch is lagged by 1. Remove it.
        csv_dict.update({k: v for k, v in metrics.items()
                         if k not in ['epoch', 'global_step']})

        result_f = pjoin(pl_module.hparams.logdir, pl_module.hparams.name,
                         'results.tsv')
        output_csv(result_f, csv_dict, delimiter='\t')

    def on_train_end(self, trainer, pl_module):
        result_f = pjoin(pl_module.hparams.logdir, pl_module.hparams.name,
                         'results.tsv')
        assert pexists(result_f), 'Result %s should exists!' % result_f

        df = pd.read_csv(result_f, delimiter='\t')
        if isinstance(pl_module, ImageNetLightningModel): # 'imageneta'
            # Use the last step as the final model!
            best_idx = -1
        else:
            func = {'min': np.argmin, 'max': np.argmax}[
                trainer.checkpoint_callback.mode]
            best_idx = int(func(df[trainer.checkpoint_callback.monitor].values))

        best_metric = df.iloc[best_idx].to_dict()

        csv_dict = OrderedDict()
        csv_dict['name'] = pl_module.hparams.name
        csv_dict.update(best_metric)
        csv_dict.update(
            vars(pl_module.hparams) if isinstance(pl_module.hparams, Namespace)
            else pl_module.hparams)

        postfix = '_test' if pl_module.hparams.test_run else ''
        dr_exp = ''
        if pl_module.is_data_ratio_exp():
            dr_exp += 'dr_'
        if pl_module.is_bbox_noise_exp():
            dr_exp += 'bn_'
        fname = pjoin(pl_module.hparams.result_dir,
                      f'{pl_module.__class__.__name__}_{dr_exp}results{postfix}.tsv')

        # Check if already exists
        if not pexists(fname):
            output_csv(fname, csv_dict, delimiter='\t')
        else:
            try:
                tmp_df = pd.read_csv(fname, delimiter='\t')
                if pl_module.hparams.name not in set(tmp_df['name']):
                    output_csv(fname, csv_dict, delimiter='\t')
            except:
                # Avoid reading CSV error
                output_csv(fname, csv_dict, delimiter='\t')

        bpath = pjoin(pl_module.hparams.logdir, pl_module.hparams.name, 'best.ckpt')
        if pexists(bpath):
            return

        if os.path.islink(bpath):
            os.unlink(bpath)

        if isinstance(pl_module, ImageNetLightningModel): # 'imageneta'
            os.symlink('last.ckpt', bpath)
        else:
            best_filename = trainer.checkpoint_callback.format_checkpoint_name(
                best_metric['epoch'], dict(gstep=best_metric['global_step']))
            os.symlink(best_filename, bpath)

    def on_test_start(self, trainer, pl_module):
        # Check if it already runs
        dr_exp = ''
        if pl_module.is_data_ratio_exp():
            dr_exp += 'dr_'
        if pl_module.is_bbox_noise_exp():
            dr_exp += 'bn_'
        ood_fname = pjoin(pl_module.hparams.result_dir,
                          f'{pl_module.__class__.__name__}_{dr_exp}ood_results.tsv')
        if pexists(ood_fname):
            ood_df = pd.read_csv(ood_fname, delimiter='\t')
            if pl_module.hparams.name in set(ood_df['name']):
                print('Already ood test for %s. Exit!' % pl_module.hparams.name)
                sys.exit()

        # For OOD detections
        bpath = pjoin(pl_module.hparams.logdir, pl_module.hparams.name, 'best.ckpt')
        assert pexists(bpath), 'Best path %s not exists!' % bpath

        best_pl_module = pl_module.load_from_checkpoint(bpath)
        pl_module.model.load_state_dict(best_pl_module.model.state_dict())
        pl_module.current_epoch = best_pl_module.current_epoch
        pl_module.global_step = best_pl_module.global_step
        print('Load best model from %s' % bpath)

        trainer.callback_metrics = {}
        print('Clean up the callback metrics before test; Remove last val metrics')

    def on_test_end(self, trainer, pl_module):
        # For OOD detections
        metrics = trainer.callback_metrics

        csv_dict = OrderedDict()
        csv_dict['name'] = pl_module.hparams.name
        csv_dict['epoch'] = pl_module.current_epoch
        csv_dict['global_step'] = pl_module.global_step
        # In the metrics csv, the epoch is lagged by 1. Remove it.
        csv_dict.update({k: v for k, v in metrics.items()
                         if k not in ['epoch', 'global_step']})

        postfix = '_test' if pl_module.hparams.test_run else ''
        dr_exp = ''
        if pl_module.is_data_ratio_exp():
            dr_exp += 'dr_'
        if pl_module.is_bbox_noise_exp():
            dr_exp += 'bn_'

        fname = pjoin(pl_module.hparams.result_dir,
                      f'{pl_module.__class__.__name__}_{dr_exp}ood_results{postfix}.tsv')
        output_csv(fname, csv_dict, delimiter='\t')


class CSVRecording2Callback(pl.Callback):
    '''
    Not storing the validation set. So what we do is we record thing
    on test end. Now only use for Xray.
    '''
    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics

        csv_dict = OrderedDict()
        csv_dict['epoch'] = pl_module.current_epoch
        csv_dict['global_step'] = pl_module.global_step
        # In the metrics csv, the epoch is lagged by 1. Remove it.
        csv_dict.update({k: v for k, v in metrics.items()
                         if k not in ['epoch', 'global_step']})

        result_f = pjoin(pl_module.hparams.logdir, pl_module.hparams.name,
                         'results.tsv')
        output_csv(result_f, csv_dict, delimiter='\t')

    def on_test_start(self, trainer, pl_module):
        result_f = pjoin(pl_module.hparams.logdir, pl_module.hparams.name,
                         'results.tsv')
        if not pexists(result_f):
            raise Exception('WIERD!!!! No results.tsv found in the model directory.')

        bpath = pjoin(pl_module.hparams.logdir, pl_module.hparams.name, 'best.ckpt')
        if not pexists(bpath):
            df = pd.read_csv(result_f, delimiter='\t')

            func = {'min': np.argmin, 'max': np.argmax}[
                trainer.checkpoint_callback.mode]
            best_idx = int(func(df[trainer.checkpoint_callback.monitor].values))
            best_record = df.iloc[best_idx].to_dict()
            best_epoch = best_record['epoch']

            best_filename = trainer.checkpoint_callback.format_checkpoint_name(
                best_epoch, best_record)

            os.symlink(best_filename, bpath)
        assert pexists(bpath), 'Best model %s does not exist!' % bpath

        best_pl_module = pl_module.load_from_checkpoint(bpath)
        pl_module.model.load_state_dict(best_pl_module.model.state_dict())
        pl_module.current_epoch = best_pl_module.current_epoch
        pl_module.global_step = best_pl_module.global_step
        print('Load best model from %s' % bpath)

    def on_test_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics

        csv_dict = OrderedDict()
        csv_dict['name'] = pl_module.hparams.name
        csv_dict['epoch'] = pl_module.current_epoch
        csv_dict['global_step'] = pl_module.global_step
        # In the metrics csv, the epoch is lagged by 1. Remove it.
        csv_dict.update({k: v for k, v in metrics.items()
                         if k not in ['epoch', 'global_step']})

        postfix = '_test' if pl_module.hparams.test_run else ''
        fname = pjoin(pl_module.hparams.result_dir,
                      f'{pl_module.__class__.__name__}_results{postfix}.tsv')
        output_csv(fname, csv_dict, delimiter='\t')
