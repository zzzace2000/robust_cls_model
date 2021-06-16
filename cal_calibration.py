import torchxrayvision as xrv
import torchvision as tv
import os
from os.path import join as pjoin
import vis_utils
import matplotlib.pyplot as plt
import torch
from arch.lightning.xray import XRayLightningModel
from arch.lightning.base_imagenet import ImageNetLightningModel
import pandas as pd
import json
from arch.lightning.cct import CCTLightningModel
import torch.nn.functional as F
import numpy as np
from sklearn.calibration import calibration_curve
from arch.utils import output_csv
from collections import OrderedDict
from argparse import ArgumentParser, Namespace


parser = ArgumentParser(description="Fine-tune BiT-M model.")
parser.add_argument('--n_bins', type=int, default=10)
parser.add_argument('--val_batch', type=int, default=64)
parser.add_argument('--model_dir', type=str, default='./models/')

args = parser.parse_args()

# for model_dir in os.listdir(args.model_dir):
for model_dir in [
    '0812_cct_none_bbox_f1_reg1e-4_grad_y_test',
    '0812_cct_none_bbox_f1_reg1e-3_grad_y_test',
    '0812_cct_none_bbox_f1_reg0.01_grad_y_test',
    '0812_cct_none_bbox_f1_reg0.1_grad_y_test',
    '0812_cct_none_bbox_f1_reg1_grad_y_test',
    '0812_cct_none',
    '0812_cct_mean',
    '0812_cct_random',
    '0812_cct_shuffle',
]:
    hparams = json.load(open(pjoin(model_dir, 'hparams.json')))

    # Get the best model. But sometimes there is bug
    df = pd.read_csv(pjoin(model_dir, 'results.csv'))

    model = eval(hparams['pl_model']).load_from_checkpoint(
        pjoin(model_dir, 'epoch=%d.ckpt' % df['val_aupr'].argmax()))

    model.hparams.val_batch = args.val_batch
    loaders = model.val_dataloader()

    model.cuda()

    with torch.no_grad():
        outputs = []

        dataloader_idx = 0
        for bi, batch in enumerate(loaders[dataloader_idx]):
            x, y = batch
            if isinstance(x, dict):
                x = x['imgs']

            x, y = x.cuda(), y.cuda()

            logit = model(x)
            prefix = ['val', 'test', 'nih', 'mimic', 'cheX'][dataloader_idx]

            output = {
                f'{prefix}_logit': logit,
                f'{prefix}_y': y,
            }
            outputs.append(output)

    logit = torch.cat([o[f'{prefix}_logit'] for o in outputs])
    y = torch.cat([o[f'{prefix}_y'] for o in outputs])
    y_onehot = torch.nn.functional.one_hot(y, num_classes=logit.shape[1])

    prob = F.softmax(logit, dim=1)

    all_y = y_onehot.reshape(-1).cpu().numpy()
    all_prob = prob.reshape(-1).cpu().numpy()

    fraction_of_positives, mean_predicted_value = \
        calibration_curve(all_y, all_prob, n_bins=args.n_bins)
    hist, bins = np.histogram(all_prob, bins=args.n_bins)

    result = OrderedDict()
    result['name'] = hparams['name']
    result['fraction_of_positives'] = fraction_of_positives.tolist()
    result['mean_predicted_value'] = mean_predicted_value.tolist()
    result['hist'] = hist.tolist()
    result['bins'] = bins.tolist()
    result.update(hparams)

    output_csv('./results/%s_calibration.tsv' % hparams['pl_model'],
               result, delimiter='\t')
