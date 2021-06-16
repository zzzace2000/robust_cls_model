# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Fine-tune a BiT model on some downstream dataset."""
#!/usr/bin/env python3
# coding: utf-8
from os.path import join as pjoin  # pylint: disable=g-importing-member
import os
import time
import random

import numpy as np
import torch
import torchvision as tv
import json
import pandas as pd
from apex import amp
import apex
import torchvision

import bit_pytorch.fewshot as fs
import bit_pytorch.lbtoolbox as lb
import bit_pytorch.models as models

import bit_common
import bit_hyperrule
import models_utils
import data_utils
from data_utils import ImagenetBoundingBoxFolder, bbox_collate, Sample
import torch.nn.functional as F
from arch.Inpainting.Baseline import RandomColorWithNoiseInpainter, LocalMeanInpainter
from arch.Inpainting.CAInpainter import CAInpainter


def topk(output, target, ks=(1,)):
  """Returns one boolean vector for each k, whether the target is within the output's top-k."""
  _, pred = output.topk(max(ks), 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))
  return [correct[:k].max(0)[0] for k in ks]


def recycle(iterable):
  """Variant of itertools.cycle that does not save iterates."""
  while True:
    for i in iterable:
      yield i


def mktrainval(args, logger):
  if args.dataset not in ['objectnet', 'imageneta'] or args.inpaint == 'none':
    return _mktrainval(args, logger)
  else:
    logger.info(f"Composing 2 loaders for {args.dataset} w/ inpaint {args.inpaint}")
    # Compose 2 loaders: 1 w/ inpaint as true and dataset == 'objectnet_bbox'
    # The other would be having 1 w/ inpaint == 'none' and dataset == 'objectnet_no_bbox'
    orig_inpaint = args.inpaint
    orig_dataset = args.dataset

    args.dataset = f'{orig_dataset}_bbox'
    f_n_train, n_classes, f_train_loader, valid_loader = mktrainval(args, logger)
    args.dataset = f'{orig_dataset}_no_bbox'
    args.inpaint = 'none'
    s_n_train, _, s_train_loader, _ = mktrainval(args, logger)

    n_train = f_n_train + s_n_train
    def composed_train_loader():
      loaders = [f_train_loader, s_train_loader]
      order = np.random.randint(low=0, high=2)
      for s in loaders[order]:
        yield s
      logger.info(f"Finish the {order} loader. (0 means bbox, 1 means no bbox)")
      for s in loaders[1 - order]:
        yield s
      logger.info(f"Finish the {1 - order} loader. (0 means bbox, 1 means no bbox)")

    train_loader = composed_train_loader()

    # Set everything back
    args.dataset = orig_dataset
    args.inpaint = orig_inpaint
    logger.info(f"Using a total training set {n_train} images")
    return n_train, n_classes, train_loader, valid_loader


def _mktrainval(args, logger):
  """Returns train and validation datasets."""
  precrop, crop = bit_hyperrule.get_resolution_from_dataset(args.dataset)
  if args.test_run: # save memory
    precrop, crop = 64, 56

  train_tx = tv.transforms.Compose([
      tv.transforms.Resize((precrop, precrop)),
      tv.transforms.RandomCrop((crop, crop)),
      tv.transforms.RandomHorizontalFlip(),
      tv.transforms.ToTensor(),
      tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])
  val_tx = tv.transforms.Compose([
      tv.transforms.Resize((crop, crop)),
      tv.transforms.ToTensor(),
      tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])

  collate_fn = None
  n_train = None
  micro_batch_size = args.batch // args.batch_split
  if args.dataset == "cifar10":
    train_set = tv.datasets.CIFAR10(args.datadir, transform=train_tx, train=True, download=True)
    valid_set = tv.datasets.CIFAR10(args.datadir, transform=val_tx, train=False, download=True)
  elif args.dataset == "cifar100":
    train_set = tv.datasets.CIFAR100(args.datadir, transform=train_tx, train=True, download=True)
    valid_set = tv.datasets.CIFAR100(args.datadir, transform=val_tx, train=False, download=True)
  elif args.dataset == "imagenet2012":
    train_set = tv.datasets.ImageFolder(pjoin(args.datadir, "train"), transform=train_tx)
    valid_set = tv.datasets.ImageFolder(pjoin(args.datadir, "val"), transform=val_tx)
  elif args.dataset.startswith('objectnet') or args.dataset.startswith('imageneta'): # objectnet and objectnet_bbox and objectnet_no_bbox
    identifier = 'objectnet' if args.dataset.startswith('objectnet') else 'imageneta'
    valid_set = tv.datasets.ImageFolder(f"../datasets/{identifier}/", transform=val_tx)

    if args.inpaint == 'none':
      if args.dataset == 'objectnet' or args.dataset == 'imageneta':
        train_set = tv.datasets.ImageFolder(pjoin(args.datadir, f"train_{args.dataset}"),
                                            transform=train_tx)
      else: # For only images with or w/o bounding box
        train_bbox_file = '../datasets/imagenet/LOC_train_solution_size.csv'
        df = pd.read_csv(train_bbox_file)
        filenames = set(df[df.bbox_ratio <= args.bbox_max_ratio].ImageId)
        if args.dataset == f"{identifier}_no_bbox":
          is_valid_file = lambda path: os.path.basename(path).split('.')[0] not in filenames
        elif args.dataset == f"{identifier}_bbox":
          is_valid_file = lambda path: os.path.basename(path).split('.')[0] in filenames
        else:
          raise NotImplementedError()

        train_set = tv.datasets.ImageFolder(
          pjoin(args.datadir, f"train_{identifier}"),
          is_valid_file=is_valid_file,
          transform=train_tx)
    else: # do inpainting
      train_tx = tv.transforms.Compose([
        data_utils.Resize((precrop, precrop)),
        data_utils.RandomCrop((crop, crop)),
        data_utils.RandomHorizontalFlip(),
        data_utils.ToTensor(),
        data_utils.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
      ])

      train_set = ImagenetBoundingBoxFolder(
        root=f"../datasets/imagenet/train_{identifier}",
        bbox_file='../datasets/imagenet/LOC_train_solution.csv',
        transform=train_tx)
      collate_fn = bbox_collate
      n_train = len(train_set) * 2
      micro_batch_size //= 2

  else:
    raise ValueError(f"Sorry, we have not spent time implementing the "
                     f"{args.dataset} dataset in the PyTorch codebase. "
                     f"In principle, it should be easy to add :)")

  if args.examples_per_class is not None:
    logger.info(f"Looking for {args.examples_per_class} images per class...")
    indices = fs.find_fewshot_indices(train_set, args.examples_per_class)
    train_set = torch.utils.data.Subset(train_set, indices=indices)

  logger.info(f"Using a training set with {len(train_set)} images.")
  logger.info(f"Using a validation set with {len(valid_set)} images.")

  valid_loader = torch.utils.data.DataLoader(
      valid_set, batch_size=micro_batch_size, shuffle=False,
      num_workers=args.workers, pin_memory=True, drop_last=False)

  if micro_batch_size <= len(train_set):
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=micro_batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=False,
        collate_fn=collate_fn)
  else:
    # In the few-shot cases, the total dataset size might be smaller than the batch-size.
    # In these cases, the default sampler doesn't repeat, so we need to make it do that
    # if we want to match the behaviour from the paper.
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=micro_batch_size, num_workers=args.workers, pin_memory=True,
        sampler=torch.utils.data.RandomSampler(train_set, replacement=True, num_samples=micro_batch_size),
        collate_fn=collate_fn)

  if n_train is None:
    n_train = len(train_set)
  return n_train, len(valid_set.classes), train_loader, valid_loader


def run_eval(model, data_loader, device, chrono, logger, step):
  # switch to evaluate mode
  model.eval()

  logger.info("Running validation...")
  logger.flush()

  all_c, all_top1, all_top5 = [], [], []
  end = time.time()
  for b, (x, y) in enumerate(data_loader):
    with torch.no_grad():
      x = x.to(device, non_blocking=True)
      y = y.to(device, non_blocking=True)

      # measure data loading time
      chrono._done("eval load", time.time() - end)

      # compute output, measure accuracy and record loss.
      with chrono.measure("eval fprop"):
        logits = model(x)
        c = torch.nn.CrossEntropyLoss(reduction='none')(logits, y)
        top1, top5 = topk(logits, y, ks=(1, 5))
        all_c.extend(c.cpu())  # Also ensures a sync point.
        all_top1.extend(top1.cpu())
        all_top5.extend(top5.cpu())

    # measure elapsed time
    end = time.time()

  model.train()
  logger.info(f"Validation@{step} loss {np.mean(all_c):.5f}, "
              f"top1 {np.mean(all_top1):.2%}, "
              f"top5 {np.mean(all_top5):.2%}")
  logger.flush()
  return all_c, all_top1, all_top5


def mixup_data(x, y, l):
  """Returns mixed inputs, pairs of targets, and lambda"""
  indices = torch.randperm(x.shape[0]).to(x.device)

  mixed_x = l * x + (1 - l) * x[indices]
  y_a, y_b = y, y[indices]
  return mixed_x, y_a, y_b


def mixup_criterion(criterion, pred, y_a, y_b, l):
  return l * criterion(pred, y_a) + (1 - l) * criterion(pred, y_b)


def main(args):
  logger = bit_common.setup_logger(args)
  if args.test_run:
    args.batch = 8
    args.batch_split = 1
    args.workers = 1

  logger.info("Args: " + str(args))

  # Fix seed
  # torch.manual_seed(args.seed)
  # torch.backends.cudnn.deterministic = True
  # torch.backends.cudnn.benchmark = False
  # np.random.seed(args.seed)
  # random.seed(args.seed)

  # Speed up
  torch.backends.cudnn.banchmark = True

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  logger.info(f"Going to train on {device}")

  n_train, n_classes, train_loader, valid_loader = mktrainval(args, logger)

  if args.inpaint != 'none':
    if args.inpaint == 'mean':
      inpaint_model = (lambda x, mask: x*mask)
    elif args.inpaint == 'random':
      inpaint_model = RandomColorWithNoiseInpainter((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    elif args.inpaint == 'local':
      inpaint_model = LocalMeanInpainter(window=)
    elif args.inpaint == 'cagan':
      inpaint_model = CAInpainter(
        valid_loader.batch_size, checkpoint_dir='./inpainting_models/release_imagenet_256/')
    else:
      raise NotImplementedError(f"Unkown inpaint {args.inpaint}")


  logger.info(f"Training {args.model}")
  if args.model in models.KNOWN_MODELS:
    model = models.KNOWN_MODELS[args.model](head_size=n_classes, zero_head=True)
  else: # from torchvision
    model = getattr(torchvision.models, args.model)(pretrained=args.finetune)

  # Resume fine-tuning if we find a saved model.
  step = 0

  # Optionally resume from a checkpoint.
  # Load it to CPU first as we'll move the model to GPU later.
  # This way, we save a little bit of GPU memory when loading.
  savename = pjoin(args.logdir, args.name, "model.pt")
  try:
    logger.info(f"Model will be saved in '{savename}'")
    checkpoint = torch.load(savename, map_location="cpu")
    logger.info(f"Found saved model to resume from at '{savename}'")

    step = checkpoint["step"]
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)

    # Note: no weight-decay!
    optim = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    optim.load_state_dict(checkpoint["optim"])
    logger.info(f"Resumed at step {step}")
  except FileNotFoundError:
    if args.finetune:
      logger.info("Fine-tuning from BiT")
      model.load_from(np.load(f"models/{args.model}.npz"))

    model = model.to(device)
    optim = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

  if args.fp16:
    model, optim = amp.initialize(model, optim, opt_level="O1")

  logger.info("Moving model onto all GPUs")
  model = torch.nn.DataParallel(model)

  optim.zero_grad()

  model.train()
  mixup = 0
  if args.mixup:
    mixup = bit_hyperrule.get_mixup(n_train)

  cri = torch.nn.CrossEntropyLoss().to(device)
  def counterfact_cri(logit, y):
    if torch.all(y >= 0):
      return F.cross_entropy(logit, y, reduction='mean')

    loss1 = F.cross_entropy(logit[y >= 0], y[y >= 0], reduction='sum')

    cf_logit, cf_y = logit[y < 0], -(y[y < 0] + 1)

    # Implement my own logsumexp trick
    m, _ = torch.max(cf_logit, dim=1, keepdim=True)
    exp_logit = torch.exp(cf_logit - m)
    sum_exp_logit = torch.sum(exp_logit, dim=1)

    eps = 1e-20
    num = (sum_exp_logit - exp_logit[torch.arange(exp_logit.shape[0]), cf_y])
    num = torch.log(num + eps)
    denon = torch.log(sum_exp_logit + eps)

    # Negative log probability
    loss2 = -(num - denon).sum()
    return (loss1 + loss2) / y.shape[0]

  logger.info("Starting training!")
  chrono = lb.Chrono()
  accum_steps = 0
  mixup_l = np.random.beta(mixup, mixup) if mixup > 0 else 1
  end = time.time()

  with lb.Uninterrupt() as u:
    for x, y in recycle(train_loader):
      # measure data loading time, which is spent in the `for` statement.
      chrono._done("load", time.time() - end)

      if u.interrupted:
        break

      # Handle inpainting
      if not isinstance(x, Sample) or x.bbox == [None] * len(x.bbox):
        criteron = cri
      else:
        criteron = counterfact_cri

        bboxes = x.bbox
        x = x.img

        # is_bbox_exists = x.new_ones(x.shape[0], dtype=torch.bool)
        mask = x.new_ones(x.shape[0], 1, *x.shape[2:])
        for i, bbox in enumerate(bboxes):
          for coord_x, coord_y, w, h in zip(bbox.xs, bbox.ys, bbox.ws, bbox.hs):
            mask[i, 0, coord_y:(coord_y + h), coord_x:(coord_x + w)] = 0.

        impute_x = inpaint_model(x, mask)
        impute_y = (-y - 1)

        x = torch.cat([x, impute_x], dim=0)
        # label -1 as negative of class 0, -2 as negative of class 1 etc...
        y = torch.cat([y, impute_y], dim=0)

      # Schedule sending to GPU(s)
      x = x.to(device, non_blocking=True)
      y = y.to(device, non_blocking=True)

      # Update learning-rate, including stop training if over.
      lr = bit_hyperrule.get_lr(step, n_train, args.base_lr)
      if lr is None:
        break
      for param_group in optim.param_groups:
        param_group["lr"] = lr

      if mixup > 0.0:
        x, y_a, y_b = mixup_data(x, y, mixup_l)

      # compute output
      with chrono.measure("fprop"):
        logits = model(x)
        if mixup > 0.0:
          c = mixup_criterion(criteron, logits, y_a, y_b, mixup_l)
        else:
          c = criteron(logits, y)
        c_num = float(c.data.cpu().numpy())  # Also ensures a sync point.

      # Accumulate grads
      with chrono.measure("grads"):
        loss = (c / args.batch_split)
        if args.fp16:
          with amp.scale_loss(loss, optim) as scaled_loss:
            scaled_loss.backward()
        else:
          loss.backward()
        accum_steps += 1

      accstep = f" ({accum_steps}/{args.batch_split})" if args.batch_split > 1 else ""
      logger.info(f"[step {step}{accstep}]: loss={c_num:.5f} (lr={lr:.1e})")  # pylint: disable=logging-format-interpolation
      logger.flush()

      # Update params
      if accum_steps == args.batch_split:
        with chrono.measure("update"):
          optim.step()
          optim.zero_grad()
        step += 1
        accum_steps = 0
        # Sample new mixup ratio for next batch
        mixup_l = np.random.beta(mixup, mixup) if mixup > 0 else 1

        # Run evaluation and save the model.
        if args.eval_every and step % args.eval_every == 0:
          run_eval(model, valid_loader, device, chrono, logger, step)
          if args.save:
            torch.save({
                "step": step,
                "model": model.module.state_dict(),
                "optim": optim.state_dict(),
            }, savename)

      end = time.time()

    # Save model!!
    if args.save:
      torch.save({
        "step": step,
        "model": model.module.state_dict(),
        "optim": optim.state_dict(),
      }, savename)

    json.dump({
      'model': args.model,
      'head_size': n_classes,
      'inpaint': args.inpaint,
      'dataset': args.dataset,
    }, open(pjoin(args.logdir, args.name, 'hyperparams.json'), 'w'))

    # Final eval at end of training.
    run_eval(model, valid_loader, device, chrono, logger, step='end')


  logger.info(f"Timings:\n{chrono}")


if __name__ == "__main__":
  parser = bit_common.argparser(models.KNOWN_MODELS.keys())
  parser.add_argument("--datadir", required=True,
                      help="Path to the ImageNet data folder, preprocessed for torchvision.")
  parser.add_argument("--workers", type=int, default=8,
                      help="Number of background threads used to load data.")
  parser.add_argument("--no-save", dest="save", action="store_false")

  # My own arguments
  parser.add_argument("--inpaint", type=str, default='none',
                      choices=['mean',  'random', 'local',
                               'cagan', 'none'])
  parser.add_argument("--bbox_subsample_ratio", type=float, default=1)
  parser.add_argument("--bbox_max_ratio", type=float, default=0.5)
  parser.add_argument("--mixup", type=int, default=0) # Turn off mixup for now
  parser.add_argument("--test_run", type=int, default=0)
  parser.add_argument("--fp16", type=int, default=1)
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--finetune", type=int, default=int)

  args = parser.parse_args()
  main(parser.parse_args())
