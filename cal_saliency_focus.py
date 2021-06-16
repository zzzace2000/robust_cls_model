import argparse
import os
from os.path import join as pjoin, exists as pexists

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from captum.attr import (
    DeepLift,
    DeepLiftShap,
    LayerGradCam,
)
from pytorch_lightning import seed_everything
from sklearn.metrics import average_precision_score, roc_auc_score

from arch.data.cct_datasets import MyCCT_Dataset
from arch.data.imagenet_datasets import MyImageFolder
from arch.data.waterbirds_datasets import WaterbirdDataset
from arch.lightning.cct import CCTLightningModel
from arch.lightning.in9 import IN9LightningModel
from arch.lightning.waterbird import WaterbirdLightningModel
from arch.utils import output_csv, generate_mask


def myclip(img):
    clip_std = img.std().item()
    img = img.clamp(min=-3*clip_std, max=3*clip_std)
    img = (1 + img / clip_std) * 0.5
    return img


def get_grad_imp(model, X, y=None, mode='grad', return_y=False, clip=False, baselines=None):
    X.requires_grad_(True)
#     X = X.cuda()

    if mode in ['grad']:
        logits = model(X)
        if y is None:
            y = logits.argmax(dim=1)

        attributions = torch.autograd.grad(
            logits[torch.arange(len(logits)), y].sum(), X)[0].detach()
    else:
        if y is None:
            with torch.no_grad():
                logits = model(X)
                y = logits.argmax(dim=1)

        if mode == 'deeplift':
            dl = DeepLift(model)

            attributions = dl.attribute(inputs=X, baselines=0., target=y)
            attributions = attributions.detach()
#             attributions = (attributions.detach() ** 2).sum(dim=1, keepdim=True)
#         elif mode in ['deepliftshap', 'deepliftshap_mean']:
        elif mode in ['deepliftshap']:
            dl = DeepLiftShap(model)
            attributions = []
            for idx in range(0, len(X), 2):
                the_x, the_y = X[idx:(idx+2)], y[idx:(idx+2)]

                attribution = dl.attribute(inputs=the_x, baselines=baselines, target=the_y)
                attributions.append(attribution.detach())
            attributions = torch.cat(attributions, dim=0)
#             attributions = dl.attribute(inputs=X, baselines=baselines, target=y).detach()
#             if mode == 'deepliftshap':
#                 attributions = (attributions ** 2).sum(dim=1, keepdim=True)
#             else:
#                 attributions = (attributions).mean(dim=1, keepdim=True)
        elif mode in ['gradcam']:
            orig_lgc = LayerGradCam(model, model.body[0])
            attributions = orig_lgc.attribute(X, target=y)

            attributions = F.interpolate(
                attributions, size=X.shape[-2:], mode='bilinear')
        else:
            raise NotImplementedError(f'${mode} is not specified.')

    # Do clipping!
    if clip:
        attributions = myclip(attributions)

    X.requires_grad_(False)
    if not return_y:
        return attributions
    return attributions, y


def get_X_and_y_by_idxs(idxes, dataset):
    X, y = [], []
    for idx in idxes:
        s, t = dataset[idx]
        X.append(s)
        y.append(t)
    X = torch.stack(X)
    y = torch.tensor(y)
    return X, y


def get_X_and_y_mask_by_idxs(idxes, dataset):
    X, y, masks = [], [], []
    for idx in idxes:
        idx = int(idx)
        s, t = dataset[idx]
        if isinstance(s, dict): # the training data loader
            X.append(s['imgs'])
            y.append(t)
            masks.append(s['masks'])
        else:
            path, _ = dataset.samples[idx]
            tmp = path.split('/')
            cls_name = tmp[-2]
            fname = tmp[-1].split('.')[0]
            if 'fg' in fname: # 'mixed_next'
                fname = fname[3:18] + '.npy'
            else: # 'original'
                fname = fname + '.npy'

            mask_np = pjoin('../datasets/bg_challenge/test/fg_mask/val/', cls_name, fname)
            mask = torch.from_numpy(np.load(mask_np))

            X.append(s)
            y.append(t)
            masks.append(mask.unsqueeze(0))

    X = torch.stack(X)
    y = torch.tensor(y)
    masks = torch.stack(masks)
    return X, y, masks


def cal_imps_and_masks(model, dataset, batch_size=4, mode='grad', target='y_pred'):
    assert target in ['y_pred', 'y']
    if 'deeplift' in mode:
        b_loader = iter(dataset.make_loader(
            batch_size=20, shuffle=True, workers=2, drop_last=True))

    if isinstance(dataset, MyCCT_Dataset) or isinstance(dataset, WaterbirdDataset):
        def gen_loader():
            loader = dataset.make_loader(
                batch_size=batch_size, shuffle=False, workers=4)
            for s, y in loader:
                if 'masks' not in s:
                    masks = generate_mask(s['imgs'], s['xs'], s['ys'], s['ws'], s['hs'])
                else:
                    masks = s['masks']
                yield s['imgs'], y, masks
    else:
        def gen_loader():
            idxes = torch.arange(len(dataset))
            for the_idxes in torch.split(idxes, batch_size):
                X, y, mask = get_X_and_y_mask_by_idxs(the_idxes, dataset)
                yield X, y, mask

    results = {}
    results['norm_fg'] = []
    results['norm_aupr'] = []
    results['norm_auc'] = []
    results['norm_iou'] = []
    for X, y, mask in gen_loader():
        X = X.cuda()

        baselines = None
        if 'deeplift' in mode:
            baselines, _ = next(b_loader)
            if isinstance(baselines, dict):
                baselines = baselines['imgs']
            baselines = baselines.cuda()

        imp = get_grad_imp(model, X, mode=mode, baselines=baselines,
                           **({} if target == 'y_pred' else {'y': y.cuda()}))
        # norm
        imp = (imp ** 2).sum(dim=1, keepdim=True).cpu()
        results['norm_fg'] += cal_bbox_metric(imp, mask)
        results['norm_aupr'] += cal_bbox_metric(imp, mask, kind='aupr')
        results['norm_auc'] += cal_bbox_metric(imp, mask, kind='auc')
        results['norm_iou'] += cal_bbox_metric(imp, mask, kind='iou')

    results['norm_fg'] = np.mean(results['norm_fg'])
    results['norm_aupr'] = np.mean(results['norm_aupr'])
    results['norm_auc'] = np.mean(results['norm_auc'])
    results['norm_iou'] = np.mean(results['norm_iou'])
    return results


def cal_bbox_metric(imps, masks, kind='percentage'):
    if kind == 'percentage':
        orig_bbox_o = (imps * masks.float()).sum(dim=[1, 2, 3])
        orig_all = (imps).sum(dim=[1, 2, 3])

        return (orig_bbox_o / orig_all).tolist()

    assert imps.shape == masks.shape
    results = []
    for imp, mask in zip(imps, masks):
        if (mask == 1).all() or (mask == 0).all():
            continue

        if kind == 'aupr':
            results.append(average_precision_score(
                mask.cpu().view(-1).int().numpy(), imp.cpu().view(-1).numpy()))
        elif kind == 'auc':
            results.append(roc_auc_score(
                mask.cpu().view(-1).int().numpy(), imp.cpu().view(-1).numpy()))
        elif kind == 'iou':
            mask = mask.cpu().view(-1)

            imp = imp.cpu().view(-1)
            thresh = imp.kthvalue(k=(mask == 0).int().sum()).values
            imp[imp > thresh] = 1
            imp[imp <= thresh] = 0

            intersection = ((imp == 1) & (mask == 1)).float().sum().item()
            union = ((imp == 1) | (mask == 1)).float().sum().item()

            results.append(intersection * 1.0 / union)
        else:
            raise NotImplementedError()

    return results


def main():
    parser = argparse.ArgumentParser(description="Fine-tune BiT-M model.")
    parser.add_argument("--prefixes", nargs='+', type=str, required=True)
    args, _ = parser.parse_known_args()

    seed_everything(2020)

    for seed in [5, 10, 100]:
        for prefix in args.prefixes:
            name = f"{prefix}_s{seed}"

            if 'in9' in name:
                output_file = 'IN9_saliency.csv'
                the_cls_name = IN9LightningModel.__name__
            elif 'cct' in name:
                output_file = 'CCT_saliency.csv'
                the_cls_name = CCTLightningModel.__name__
            elif 'wb' in name:
                output_file = 'wb_saliency.csv'
                the_cls_name = WaterbirdLightningModel.__name__
            else:
                raise NotImplementedError()

            prev_df = None
            if pexists(f'./results/{output_file}'):
                prev_df = pd.read_csv(f'./results/{output_file}')

            model_path = f'./models/{name}/best.ckpt'
            if not pexists(model_path):
                print(f'Copying from v for {model_path}')
                os.system(f'rsync -avzL '
                          f'vr:/h/kingsley/bbox_deconv/big_transfer/models/{name}/best.ckpt ./models/{name}/')

            tmp = eval(the_cls_name).load_from_checkpoint(model_path)
            model = tmp.model
            # hparams = tmp.hparams
            model.cuda().eval()

            for p in model.parameters():
                p.requires_grad_(False)

            if 'in9' in name:
                original_d = MyImageFolder(
                    '../datasets/bg_challenge/test/original/val/',
                    MyImageFolder.get_val_transform()
                )
                mixed_same_d = MyImageFolder(
                    '../datasets/bg_challenge/test/mixed_same/val/',
                    MyImageFolder.get_val_transform()
                )
                mixed_next_d = MyImageFolder(
                    '../datasets/bg_challenge/test/mixed_next/val/',
                    MyImageFolder.get_val_transform()
                )
                arr = [
                    ('All mixed_same', mixed_same_d),
                    ('All original', original_d),
                    ('All mixed_next', mixed_next_d),
                ]
            elif 'cct' in name:
                cis_test_d = MyCCT_Dataset(
                    '../datasets/cct/eccv_18_annotation_files/cis_test_annotations.json',
                    transform=MyCCT_Dataset.get_val_bbox_transform(),
                    only_bbox_imgs=True,
                )
                trans_test_d = MyCCT_Dataset(
                    '../datasets/cct/eccv_18_annotation_files/trans_test_annotations.json',
                    transform=MyCCT_Dataset.get_val_bbox_transform(),
                    only_bbox_imgs=True,
                )
                print('cis:', len(cis_test_d))
                print('trans:', len(trans_test_d))
                arr = [
                    ('All cis_test', cis_test_d),
                    ('All trans_test', trans_test_d),
                ]
            elif 'wb' in name:
                orig_test_d = WaterbirdDataset(
                    mode='test',
                    type='same',
                    only_images=False,
                    transform=WaterbirdDataset.get_val_transform())
                flip_test_d = WaterbirdDataset(
                    mode='test',
                    type='flip',
                    only_images=False,
                    transform=WaterbirdDataset.get_val_transform())
                print('orig:', len(orig_test_d))
                print('flip:', len(flip_test_d))
                arr = [
                    ('All orig', orig_test_d),
                    ('All flip', flip_test_d),
                ]
            else:
                raise NotImplementedError('Not found %s' % name)

            for name_idxes, dataset in arr:
                for mode, bs in [
                    # ('grad', 32),
                    ('deepliftshap', 64),
                ]:
                    for target in [
                        #                 'y_pred',
                        'y',
                    ]:
                        if prev_df is not None \
                                and ((prev_df['name'] == name)
                                & (prev_df['mode'] == mode)
                                & (prev_df['name_idxes'] == name_idxes)).any():
                            continue

                        print(name, name_idxes, mode)
                        result = cal_imps_and_masks(model, dataset, mode=mode, batch_size=bs, target=target)

                        # result = {}
                        result['name_idxes'] = name_idxes

                        if 'in9' in name:
                            name_m = '_'.join(prefix.split('_')[3:])
                        else:
                            name_m = '_'.join(prefix.split('_')[2:])
                        result['name_m'] = name_m

                        result['name'] = name
                        result['seed'] = seed
                        result['mode'] = mode
                        result['target'] = target

                        output_csv(f'./results/{output_file}', result)


if __name__ == '__main__':
    main()
