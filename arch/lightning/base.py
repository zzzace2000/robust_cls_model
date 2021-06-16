from collections import OrderedDict
from collections import OrderedDict
from os.path import exists as pexists
from os.path import join as pjoin  # pylint: disable=g-importing-member

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torch.utils.data
import torch.utils.data.distributed
import torch.utils.data.distributed
from pytorch_lightning.core import LightningModule
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, roc_auc_score

from .. import bit_common
from ..data.imagenet_datasets import MySubset
from ..grid import GridMask
from ..inpainting.AdvInpainting import AdvInpainting
from ..inpainting.Baseline import RandomColorWithNoiseInpainter, ShuffleInpainter, \
    TileInpainter, FactualMixedRandomTileInpainter
from ..label_smoothing import LabelSmoothingCrossEntropy
from ..utils import DotDict, generate_mask, make_masks_as_rectangular


class BaseLightningModel(LightningModule):
    def __init__(self, hparams):
        """
        Training imagenet models by fintuning from Big-Transfer models
        """
        super().__init__()
        if isinstance(hparams, dict): # Fix the bug in pl in reloading
            hparams = DotDict(hparams)
        self.hparams = hparams

        # Backward compatability
        if 'cf' not in self.hparams:
            self.hparams.cf = 'logp'
        if 'result_dir' not in self.hparams:
            self.hparams.result_dir = './results/'
        if 'mixup' not in self.hparams:
            self.hparams.mixup = 0.
        if 'eps_label_smoothing' not in self.hparams:
            self.hparams.eps_label_smoothing = 0.
        if 'grid' not in self.hparams:
            self.hparams.grid = 0

        if self.hparams.grid > 0:
            # I set the default parameter same as the original repo
            # for ImageNet experiment
            self.grid_mask = GridMask()
        if self.hparams.cf.startswith('channels'):
            assert self.hparams.inpaint != 'none'

        self.my_logger = bit_common.setup_logger(self.hparams)
        self.train_loader = None
        self.valid_loaders = None
        self.test_loaders = None
        self.stage = 'train' # Record it's now train/val/test

        self.init_setup()

        self.inpaint = self.get_inpainting_model(self.hparams.inpaint)
        self.f_inpaint = self.get_inpainting_model(self.hparams.f_inpaint)

    def forward(self, x, return_cf_channels=False):
        if self.hparams.grid > 0:
            x = self.grid_mask(x)

        logits = self.model(x)
        if self.hparams.cf.startswith('channels') and (not return_cf_channels):
            # In validation time, ignore backgnd logits
            return logits[:, :int(logits.shape[1] / 2)]

        return logits

    def training_step(self, batch, batch_idx, is_training=True):
        # Baseline training
        if self.hparams.mixup > 0.:
            return self.mixup_training_step(batch, batch_idx)
        if self.hparams.eps_label_smoothing > 0.:
            return self.label_smooting_training_step(batch, batch_idx)

        s, l = batch

        is_dict = isinstance(s, dict)
        if not is_dict:
            x, y = s, l
        else:
            orig_x_len = len(s['imgs'])
            is_mask_rect = True
            if 'masks' in s:
                if s['masks'] is None: # no fg region
                    has_bbox = s['imgs'].new_zeros(s['imgs'].shape[0]).bool()
                else:
                    mask = s['masks']
                    has_bbox = (mask == 1).any(dim=3).any(dim=2).any(dim=1)
                    is_mask_rect = False
            else:
                has_bbox = (s['xs'] != -1)
                if has_bbox.ndim == 2: # multiple bboxes
                    has_bbox = has_bbox.any(dim=1)

                mask = generate_mask(
                    s['imgs'], s['xs'], s['ys'], s['ws'], s['hs'])
            if self.hparams.inpaint == 'none' or has_bbox.sum() == 0:
                x, y = s['imgs'], l
            else:
                if 'imgs_cf' in s:
                    impute_x = s['imgs_cf'][has_bbox]
                else:
                    # Transform mask into rectangular to do CF imputation
                    # to avoid shape info still existing in CF images
                    mask_cf = mask[has_bbox]
                    if not is_mask_rect:
                        mask_cf = make_masks_as_rectangular(mask_cf)
                    impute_x = self.inpaint(s['imgs'][has_bbox], 1 - mask_cf)
                impute_y = (-l[has_bbox] - 1)

                x = torch.cat([s['imgs'], impute_x], dim=0)
                # label -1 as negative of class 0, -2 as negative of class 1 etc...
                y = torch.cat([l, impute_y], dim=0)

            if self.hparams.f_inpaint != 'none' and has_bbox.any():
                impute_x = self.f_inpaint(
                    s['imgs'][has_bbox], mask[has_bbox], l[has_bbox])
                x = torch.cat([x, impute_x], dim=0)
                y = torch.cat([y, l[has_bbox]], dim=0)

        if not is_dict or self.hparams.get('reg', 'none') == 'none' \
                or (is_dict and (~has_bbox).all()):
            logits = self(x, return_cf_channels=True)
            reg_loss = logits.new_tensor(0.)
        else:
            x_orig, y_orig = x[:orig_x_len], y[:orig_x_len]
            saliency_fn = eval(f"get_{self.hparams.get('reg_grad', 'grad_y')}")
            the_grad, logits = saliency_fn(x_orig, y_orig, self,
                                           is_training=is_training)
            if torch.all(the_grad == 0.):
                reg_loss = logits.new_tensor(0.)
            elif self.hparams.reg == 'gs':
                assert is_dict and self.hparams.inpaint != 'none'
                if not has_bbox.any():
                    reg_loss = logits.new_tensor(0.)
                else:
                    x_orig, x_cf = x[:orig_x_len], x[orig_x_len:(orig_x_len + has_bbox.sum())]
                    dist = x_orig[has_bbox] - x_cf
                    cos_sim = self.my_cosine_similarity(the_grad, dist)

                    reg_loss = (1. - cos_sim).mean().mul_(self.hparams.reg_coeff)
            elif self.hparams.reg == 'bbox_o':
                reg_loss = ((the_grad[has_bbox] * (1 - mask[has_bbox])) ** 2).mean()\
                    .mul_(self.hparams.reg_coeff)
            elif self.hparams.reg == 'bbox_f1':
                norm = (the_grad[has_bbox] ** 2).sum(dim=1, keepdim=True)
                norm = (norm - norm.min()) / norm.max()
                gnd_truth = mask[has_bbox]

                f1 = self.diff_f1_score(norm, gnd_truth)
                # (1 - F1) as the loss
                reg_loss = (1. - f1).mul_(self.hparams.reg_coeff)
            else:
                raise NotImplementedError(self.hparams.reg)

            # Doing annealing for reg loss
            if self.hparams.reg_anneal > 0.:
                anneal = self.global_step / (self.hparams.max_epochs * len(self.train_loader)
                                             * self.hparams.reg_anneal)
                reg_loss *= anneal

            if len(x) > orig_x_len: # Other f or cf images
                cf_logits = self(x[orig_x_len:], return_cf_channels=True)
                logits = torch.cat([logits, cf_logits], dim=0)

        c, c_cf = self.counterfact_cri(logits, y)
        c_cf *= self.hparams.cf_coeff

        # Check NaN
        for name, metric in [
            ('train_loss', c),
            ('cf_loss', c_cf),
            ('reg_loss', reg_loss),
        ]:
            if torch.isnan(metric).all():
                raise RuntimeError(f'metric {name} is Nan')

        tqdm_dict = {'train_loss': c,
                     'cf_loss': c_cf.detach(),
                     'reg_loss': reg_loss.detach()}
        output = OrderedDict({
            'loss': c + c_cf + reg_loss,
            'train_loss': c.detach(),
            'cf_loss': c_cf.detach(),
            'reg_loss': reg_loss.detach(),
            'progress_bar': tqdm_dict,
            'log': tqdm_dict,
        })
        return output

    def mixup_training_step(self, batch, batch_idx):
        x, y = batch
        if isinstance(x, dict):
            x = x['imgs']

        mixup_l = np.random.beta(self.hparams.mixup, self.hparams.mixup)
        x, y_a, y_b = self.mixup_data(x, y, mixup_l)

        criteria = torch.nn.CrossEntropyLoss(reduction='mean')
        logits = self(x)
        c = self.mixup_criterion(criteria, logits, y_a, y_b, mixup_l)

        # Check NaN
        if torch.isnan(c).all():
            raise RuntimeError(f'metric train_loss is Nan')

        tqdm_dict = {'train_loss': c.detach()}
        output = OrderedDict({
            'loss': c,
            'train_loss': c.detach(),
            'progress_bar': tqdm_dict,
            'log': tqdm_dict,
        })
        return output

    def label_smooting_training_step(self, batch, batch_idx):
        x, y = batch
        if isinstance(x, dict):
            x = x['imgs']

        criteria = LabelSmoothingCrossEntropy(
            epsilon=self.hparams.eps_label_smoothing)

        logits = self(x)
        c = criteria(logits, y)

        if torch.isnan(c).all():
            raise RuntimeError(f'metric train_loss is Nan')

        tqdm_dict = {'train_loss': c.detach()}
        output = OrderedDict({
            'loss': c,
            'train_loss': c.detach(),
            'progress_bar': tqdm_dict,
            'log': tqdm_dict,
        })
        return output

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        if isinstance(x, dict):
            x = x['imgs']

        logit = self(x)
        prefix = self.val_sets_names[dataloader_idx]

        output = OrderedDict({
            f'{prefix}_logit': logit,
            f'{prefix}_y': y,
        })
        return output

    def validation_epoch_end(self, outputs):
        tqdm_dict = {}
        calibration_dict = {}

        def cal_metrics(output, prefix='val'):
            logit = torch.cat([o[f'{prefix}_logit'] for o in output])
            y = torch.cat([o[f'{prefix}_y'] for o in output])
            tqdm_dict[f'{prefix}_loss'] = F.cross_entropy(logit, y, reduction='mean').item()
            tqdm_dict[f'{prefix}_acc1'], = self.accuracy(logit, y, topk=(1,))

            assert logit.shape[1] > 1, 'Logits shape is wierd: ' + str(logit.shape)
            if logit.shape[1] == 2: # binary classification
                prob = F.softmax(logit, dim=1)
                y, logit, prob = y.cpu().numpy(), \
                             (logit[:, 1] - logit[:, 0]).cpu().numpy(), \
                             prob[:, 1].cpu().numpy()
                try:
                    tqdm_dict[f'{prefix}_auc'] = roc_auc_score(
                        y, logit) * 100
                    tqdm_dict[f'{prefix}_aupr'] = average_precision_score(
                        y, logit) * 100
                    fraction_of_positives, mean_predicted_value = \
                        calibration_curve(y, prob, n_bins=10)
                    tqdm_dict[f'{prefix}_ece'] = np.mean(np.abs(
                        fraction_of_positives - mean_predicted_value)) * 100
                    hist, bins = np.histogram(prob, bins=10)
                    calibration_dict[f'{prefix}_frp'] = fraction_of_positives.tolist()
                    calibration_dict[f'{prefix}_mpv'] = mean_predicted_value.tolist()
                    calibration_dict[f'{prefix}_hist'] = hist.tolist()
                    calibration_dict[f'{prefix}_bins'] = bins.tolist()
                except ValueError as e:  # only 1 class is present. Happens in sanity check
                    self.my_logger.warn('Only 1 class is present!\n' + str(e))
                    tqdm_dict[f'{prefix}_auc'] = -1.
                    tqdm_dict[f'{prefix}_aupr'] = -1.
                    tqdm_dict[f'{prefix}_ece'] = -1.
            else:
                # Calculate the average auc, average aupr, and average F1
                _, pred = torch.max(logit, dim=1)
                y_onehot = torch.nn.functional.one_hot(y, num_classes=logit.shape[1])

                # In CCT cis_val and trans_test, some classes do not exist
                non_zero_cls = (y_onehot.sum(dim=0) > 0)
                if not non_zero_cls.all():
                    y_onehot = y_onehot[:, non_zero_cls]
                    logit = logit[:, non_zero_cls]

                prob = F.softmax(logit, dim=1)
                y, y_onehot, logit, pred = y.cpu().numpy(), y_onehot.cpu().numpy(), \
                                           logit.cpu().numpy(), pred.cpu().numpy()

                all_y = y_onehot.reshape(-1)
                all_prob = prob.reshape(-1).cpu().numpy()

                fraction_of_positives, mean_predicted_value = \
                    calibration_curve(all_y, all_prob, n_bins=10)
                ece = np.mean(np.abs(fraction_of_positives - mean_predicted_value))

                tqdm_dict[f'{prefix}_ece'] = ece * 100
                try:
                    tqdm_dict[f'{prefix}_auc'] = roc_auc_score(
                        y_onehot, logit, multi_class='ovr') * 100
                    tqdm_dict[f'{prefix}_aupr'] = average_precision_score(
                        y_onehot, logit) * 100
                except ValueError as e:  # only 1 class is present. Happens in sanity check
                    self.my_logger.warn('Only 1 class is present!\n' + str(e))
                    tqdm_dict[f'{prefix}_auc'] = -1.
                    tqdm_dict[f'{prefix}_aupr'] = -1.

                hist, bins = np.histogram(all_prob, bins=10)
                calibration_dict[f'{prefix}_frp'] = fraction_of_positives.tolist()
                calibration_dict[f'{prefix}_mpv'] = mean_predicted_value.tolist()
                calibration_dict[f'{prefix}_hist'] = hist.tolist()
                calibration_dict[f'{prefix}_bins'] = bins.tolist()

        if isinstance(outputs[0], dict): # Only one val loader
            cal_metrics(outputs, self.val_sets_names[0])
        else:
            for i, n in enumerate(self.val_sets_names):
                cal_metrics(outputs[i], n)

        result = {
            'progress_bar': tqdm_dict, 'log': tqdm_dict,
            'val_loss': tqdm_dict["val_loss"],
            **calibration_dict
        }
        return result

    @staticmethod
    def my_cosine_similarity(t1, t2, eps=1e-8):
        if torch.all(t1 == 0.) or torch.all(t2 == 0.):
            return t1.new_tensor(0.)
        other_dim = list(range(1, t1.ndim))
        iprod = (t1 * t2).sum(dim=other_dim)
        t1_norm = (t1 * t1).sum(dim=other_dim).sqrt()
        t2_norm = (t2 * t2).sum(dim=other_dim).sqrt()
        cos_sim = iprod / (t1_norm * t2_norm + eps)
        return cos_sim

    @staticmethod
    def diff_f1_score(pred, gnd_truth):
        TP = pred.mul(gnd_truth).sum(dim=list(range(1, pred.ndim)))
        FP = pred.mul(1. - gnd_truth).sum(dim=list(range(1, pred.ndim)))
        FN = (1. - pred).mul(gnd_truth).sum(dim=list(range(1, pred.ndim)))

        # (1 - F1) as the loss
        return (2 * TP / (2 * TP + FP + FN)).mean()

    @classmethod
    def accuracy(cls, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def configure_optimizers(self):
        raise NotImplementedError()

    def counterfact_cri(self, logit, y):
        '''
        :return: (avg normal loss, avg ce_loss)
        '''
        zero_tensor = logit.new_tensor(0.)
        if torch.all(y >= 0):
            return F.cross_entropy(logit, y, reduction='mean'), zero_tensor
        if torch.all(y < 0):
            return zero_tensor, self.counterfactual_ce_loss(logit, y, reduction='mean')

        loss1 = F.cross_entropy(logit[y >= 0], y[y >= 0], reduction='sum')
        loss2 = self.counterfactual_ce_loss(logit[y < 0], y[y < 0], reduction='sum')

        return loss1 / logit.shape[0], loss2 / logit.shape[0]

    def counterfactual_ce_loss(self, logit, y, reduction='none'):
        assert (y < 0).all(), str(y)

        if self.hparams.cf.startswith('channels'):
            new_y = ((-y - 1) + (logit.shape[1] // 2))
            return F.cross_entropy(logit, new_y, reduction=reduction)

        cf_y = -(y + 1)
        if self.hparams.cf == 'uni': # uniform prob
            loss = -F.log_softmax(logit, dim=1).mean(dim=1)
        elif self.hparams.cf == 'uni_e': # uniform prob except the cls
            logp = F.log_softmax(logit, dim=1)
            weights = torch.ones_like(logp).mul_(1. / (
                    logp.shape[1] - 1))
            weights[torch.arange(len(cf_y)), cf_y] = 0.
            loss = -(weights * logp).sum(dim=1)
        elif self.hparams.cf == 'logp':
            if logit.shape[1] == 2: # 2-cls
                return F.cross_entropy(logit, 1 - cf_y, reduction=reduction)

            # Implement my own logsumexp trick
            m, _ = torch.max(logit, dim=1, keepdim=True)
            exp_logit = torch.exp(logit - m)
            sum_exp_logit = torch.sum(exp_logit, dim=1)

            eps = 1e-20
            num = (sum_exp_logit - exp_logit[torch.arange(exp_logit.shape[0]), cf_y])
            num = torch.log(num + eps)
            denon = torch.log(sum_exp_logit + eps)

            # Negative log probability
            loss = -(num - denon)
        else:
            raise NotImplementedError(str(self.hparams.cf))

        if reduction == 'none':
            return loss
        if reduction == 'mean':
            return loss.mean()
        if reduction == 'sum':
            return loss.sum()

    def train_dataloader(self):
        if self.train_loader is None:
            self._setup_loaders()
        return self.train_loader

    def val_dataloader(self):
        if self.valid_loaders is None:
            self._setup_loaders()
        return self.valid_loaders

    @staticmethod
    def mixup_data(x, y, l):
        """Returns mixed inputs, pairs of targets, and lambda"""
        indices = torch.randperm(x.shape[0]).to(x.device)

        mixed_x = l * x + (1 - l) * x[indices]
        y_a, y_b = y, y[indices]
        return mixed_x, y_a, y_b

    @staticmethod
    def mixup_criterion(criterion, pred, y_a, y_b, l):
        return l * criterion(pred, y_a) + (1 - l) * criterion(pred, y_b)

    def _setup_loaders(self):
        train_set, valid_sets, val_sets_names = self._make_train_val_dataset()
        self.my_logger.info(f"Using a training set with {len(train_set)} images.")
        for idx, (n, v) in enumerate(zip(val_sets_names, valid_sets)):
            self.my_logger.info(f"Using a validation set {idx} {n} with {len(v)} images.")

        self.val_sets_names = val_sets_names

        train_bs = self.hparams.batch // self.hparams.batch_split
        val_bs = train_bs
        if 'val_batch' in self.hparams:
            val_bs = self.hparams.val_batch // self.hparams.batch_split
        self.train_loader = train_set.make_loader(
            train_bs, shuffle=True, workers=self.hparams.workers)
        self.valid_loaders = [v.make_loader(
            val_bs, shuffle=False, workers=self.hparams.workers)
            for v in valid_sets]

    def _make_train_val_dataset(self):
        raise NotImplementedError()

    @staticmethod
    def sub_dataset(bbox_dataset, num_subset, sec_dataset=None):
        if sec_dataset is not None:
            assert len(sec_dataset) == len(bbox_dataset)

        if num_subset == 0:
            if sec_dataset is None:
                return None, bbox_dataset
            return None, bbox_dataset, None, sec_dataset
        if num_subset >= len(bbox_dataset):
            if sec_dataset is None:
                return bbox_dataset, None
            return bbox_dataset, None, sec_dataset, None

        tmp = torch.Generator()
        tmp.manual_seed(2020)

        indices = torch.randperm(len(bbox_dataset), generator=tmp)
        first_dataset = MySubset(bbox_dataset, indices=indices[:num_subset])
        rest_dataset = MySubset(bbox_dataset, indices=indices[num_subset:])
        if sec_dataset is None:
            return first_dataset, rest_dataset

        fs = MySubset(sec_dataset, indices=indices[:num_subset])
        rs = MySubset(sec_dataset, indices=indices[num_subset:])
        return first_dataset, rest_dataset, fs, rs

    def get_inpainting_model(self, inpaint):
        if inpaint in ['none']:
            inpaint_model = None
        elif inpaint == 'mean':
            inpaint_model = (lambda x, mask: x * mask)
        elif inpaint == 'random':
            inpaint_model = RandomColorWithNoiseInpainter((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        elif inpaint == 'shuffle':
            inpaint_model = ShuffleInpainter()
        elif inpaint == 'tile':
            inpaint_model = TileInpainter()
        elif inpaint in ['pgd', 'fgsm']:
            alpha = self.hparams.alpha
            if alpha == -1:
                alpha = self.hparams.eps * 1.25
            inpaint_model = AdvInpainting(
                self.model, eps=self.hparams.eps,
                alpha=alpha,
                attack=inpaint)
        elif inpaint == 'mrand':
            inpaint_model = FactualMixedRandomTileInpainter()
        else:
            raise NotImplementedError(f"Unkown inpaint {inpaint}")

        return inpaint_model

    def test_dataloader(self):
        '''
        This is for OOD detections. The first loader is the normal
        test set. And the rest of the loaders are from other datasets
        like Gaussian, Uniform, CCT and Xray.

        Gaussian, Uniform: generate the same number as test sets (45k)
        CCT: whole dataset like 45k?
        Xray: 30k?
        '''
        if self.test_loaders is None:
            test_sets, test_sets_names = self._make_test_datasets()
            if test_sets is None:
                return None

            self.test_sets_names = test_sets_names
            for idx, (n, v) in enumerate(zip(self.test_sets_names, test_sets)):
                self.my_logger.info(f"Using a test set {idx} {n} with {len(v)} images.")

            self.test_loaders = [v.make_loader(
                self.hparams.val_batch, shuffle=False, workers=self.hparams.workers)
                for v in test_sets]
        return self.test_loaders

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        if isinstance(x, dict):
            x = x['imgs']

        logits = self(x, return_cf_channels=True)
        prob = F.softmax(logits, dim=1)

        if self.hparams.cf.startswith('channels'):
            anomaly_score = prob[:, (prob.shape[1] // 2):].sum(dim=1)
        else:
            anomaly_score = 1. - (prob.max(dim=1).values)

        output = {
            # Can't return cpu or it will die in multi-gpu settings
            'anomaly_score': anomaly_score,
        }
        return output

    def test_epoch_end(self, outputs):
        tqdm_dict = {}

        def cal_metrics(the_test, the_orig, prefix='gn'):
            the_as = torch.cat([o['anomaly_score'] for o in the_test])
            orig_as = torch.cat([o['anomaly_score'] for o in the_orig])

            # 95% TPR: k is the kth-smallest element
            # I want 95% of examples to below this number
            thresh = torch.kthvalue(
                orig_as,
                k=int(np.floor(0.95 * len(orig_as)))).values
            fpr = (the_as <= thresh).float().mean().item()
            tqdm_dict[f'{prefix}_ood_fpr'] = fpr

            cat_as = torch.cat([the_as, orig_as], dim=0)
            ys = torch.cat([torch.ones(len(the_as)), torch.zeros(len(orig_as))], dim=0)

            tqdm_dict[f'{prefix}_ood_auc'] = roc_auc_score(
                ys.cpu().numpy(), cat_as.cpu().numpy())
            tqdm_dict[f'{prefix}_ood_aupr'] = average_precision_score(
                ys.cpu().numpy(), cat_as.cpu().numpy())

        for name, output in zip(self.test_sets_names[1:], outputs[1:]):
            cal_metrics(output, outputs[0], name)

        result = {
            'progress_bar': tqdm_dict, 'log': tqdm_dict,
        }
        return result

    def is_data_ratio_exp(self):
        return False

    def is_bbox_noise_exp(self):
        return False

    def _make_test_datasets(self):
        # Return None if no test set exists
        return None, None

    @classmethod
    def add_model_specific_args(cls, parser):  # pragma: no-cover
        raise NotImplementedError()

    def pl_trainer_args(self):
        raise NotImplementedError()

    @classmethod
    def is_finished_run(cls, model_dir):
        raise NotImplementedError()


class EpochBaseLightningModel(BaseLightningModel):
    @classmethod
    def is_finished_run(cls, model_dir):
        last_ckpt = pjoin(model_dir, 'last.ckpt')
        if pexists(last_ckpt):
            tmp = torch.load(last_ckpt,
                             map_location=torch.device('cpu'))
            last_epoch = tmp['epoch']
            hparams = tmp['hyper_parameters']
            if last_epoch >= hparams.max_epochs:
                print('Already finish fitting! Max %d Last %d'
                      % (hparams.max_epochs, last_epoch))
                return True

        return False
