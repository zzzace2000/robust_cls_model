import os
import torch
import copy
import time
import numpy as np


def output_csv(the_path, data_dict, order=None, delimiter=','):
    if the_path.endswith('.tsv'):
        delimiter = '\t'

    is_file_exists = os.path.exists(the_path)
    with open(the_path, 'a+') as op:
        keys = list(data_dict.keys())
        if order is not None:
            keys = order + [k for k in keys if k not in order]

        col_title = delimiter.join([str(k) for k in keys])
        if not is_file_exists:
            print(col_title, file=op)
        else:
            old_col_title = open(the_path, 'r').readline().strip()
            if col_title != old_col_title:
                old_order = old_col_title.split(delimiter)

                no_key = [k for k in old_order if k not in keys]
                if len(no_key) > 0:
                    print('The data_dict does not have the '
                          'following old keys: %s' % str(no_key))

                additional_keys = [k for k in keys if k not in old_order]
                if len(additional_keys) > 0:
                    print('WARNING! The data_dict has following additional '
                          'keys %s.' % (str(additional_keys)))
                    col_title = delimiter.join([
                        str(k) for k in old_order + additional_keys])
                    print(col_title, file=op)

                keys = old_order + additional_keys

        vals = []
        for k in keys:
            val = data_dict.get(k, -999)
            if isinstance(val, torch.Tensor) and val.ndim == 0:
                val = val.item()
            vals.append(str(val))

        print(delimiter.join(vals), file=op)


class Timer:
    def __init__(self, name, remove_start_msg=True):
        self.name = name
        self.remove_start_msg = remove_start_msg

    def __enter__(self):
        self.start_time = time.time()
        print('Run "%s".........' % self.name, end='\r' if self.remove_start_msg else '\n')

    def __exit__(self, exc_type, exc_val, exc_tb):
        time_diff = float(time.time() - self.start_time)
        time_str = '{:.1f}s'.format(time_diff) if time_diff >= 1 else '{:.0f}ms'.format(time_diff * 1000)

        print('Finish "{}" in {}'.format(self.name, time_str))


def generate_mask(imgs, xs, ys, ws, hs):
    ''' It sets the bbox region as 1, and the outside as 0 '''
    mask = imgs.new_zeros(imgs.shape[0], 1, *imgs.shape[2:])
    for i, (xs, ys, ws, hs) in enumerate(zip(xs, ys, ws, hs)):
        if xs.ndim == 0:
            if xs > 0:
                mask[i, 0, ys:(ys + hs), xs:(xs + ws)] = 1.
            continue

        for coord_x, coord_y, w, h in zip(xs, ys, ws, hs):
            if coord_x == -1:
                break
            mask[i, 0, coord_y:(coord_y + h), coord_x:(coord_x + w)] = 1.
    return mask


def make_masks_as_rectangular(masks):
    assert isinstance(masks, torch.Tensor), str(type(masks))
    assert masks.ndim == 4, str(masks.ndim)

    def extract_coordinates_from_mask(is_mask):
        # extract x, y, w, h
        is_mask = is_mask.all(dim=0)
        tmp = torch.arange(is_mask.shape[0])[is_mask.any(dim=1)]
        if len(tmp) == 0:
            return (torch.tensor(-1.)) * 4
        y, h = tmp[0], tmp[-1] - tmp[0] + 1
        tmp = torch.arange(is_mask.shape[1])[is_mask.any(dim=0)]
        x, w = tmp[0], tmp[-1] - tmp[0] + 1
        return x, y, w, h

    masks = masks.clone()
    for i, mask in enumerate(masks):
        x, y, w, h = extract_coordinates_from_mask((mask == 1))
        if x >= 0:
            masks[i, :, y:(y+h), x:(x+w)] = 1
    return masks


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict) and not isinstance(v, DotDict):
                self[k] = DotDict(v)

    def __deepcopy__(self, memo):
        return DotDict(copy.deepcopy(dict(self), memo=memo))
