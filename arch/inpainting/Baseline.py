import torch.nn as nn
import torch
import math
from ..utils import generate_mask


class InpaintTemplate(nn.Module):
    def forward(self, x, mask, y=None):
        return self.impute_missing_imgs(x, mask)

    def impute_missing_imgs(self, x, mask):
        raise NotImplementedError('Need to implement this')

    def reset(self):
        pass


class MeanInpainter(InpaintTemplate):
    '''
    Just put 0 to impute. (As grey in Imagenet)
    '''
    def impute_missing_imgs(self, x, mask):
        return x * mask

    def generate_background(self, x, mask):
        return x.new(x.shape).zero_()


class ShuffleInpainter(InpaintTemplate):
    '''
    It shuffles all the values including channels.
    '''
    def impute_missing_imgs(self, x, mask):
        background = self.generate_background(x, mask)
        return x * mask + background * (1. - mask)

    def generate_background(self, x, mask):
        idx = torch.randperm(x.nelement()).to(x.device)
        t = x.reshape(-1)[idx].reshape(x.size())
        return t


class TileInpainter(InpaintTemplate):
    '''
    Find the largest rectangular patch within mask=1 region, then tile.
    Used for CF(Tile) generation.
    '''
    def __init__(self, use_bbox_to_mask=False):
        super().__init__()
        self.use_bbox_to_mask = use_bbox_to_mask

    def impute_missing_imgs(self, x, mask):
        if (mask == 1).all():
            return x
        if (mask == 0).all():
            return x * 0.

        background, coords = self.generate_background(x, mask)
        if self.use_bbox_to_mask:
            # Fg mask is 0, bg mask is 1
            mask = 1 - generate_mask(x, *coords)

        return x * mask + background * (1. - mask)

    def generate_background(self, x, mask):
        # Find the largest rectangular with 1
        is_backgnd = (mask == 1)
        is_col_with_backgnd = is_backgnd.all(dim=-2)[:, 0, :]
        is_row_with_backgnd = is_backgnd.all(dim=-1)[:, 0, :]

        def find_largest_chunk(row):
            # does not seem to have an elegant vector way
            # write ugly for loop
            results, fg_x1, fg_x2 = [], [], []
            for r in row:
                if r.all(): # all background
                    results.append((0, len(r)))
                    fg_x1.append(-1)
                    fg_x2.append(-1)
                    continue
                if (~r).all(): # all foreground
                    results.append((-1, -1))
                    fg_x1.append(0)
                    fg_x2.append(len(r))
                    continue

                # r is a 1 dimension tensor
                tmp = torch.arange(len(r))
                true_idxes = tmp[r]
                false_idxes = tmp[~r]

                max_len, max_s, max_e = 0, -1, -1
                false_start = 0 if len(false_idxes) == 0 else false_idxes[0]
                false_end = len(r)
                while len(true_idxes) > 0:
                    s = true_idxes[0]
                    e = len(r) if len(false_idxes) == 0 else false_idxes[0]
                    if (e - s) > max_len:
                        max_len, max_s, max_e = (e - s), s, e

                    true_idxes = true_idxes[true_idxes > e]
                    if len(true_idxes) == 0:
                        break
                    if (false_idxes < true_idxes[0]).all():
                        false_end = true_idxes[0]
                    false_idxes = false_idxes[false_idxes > true_idxes[0]]

                results.append((max_s, max_e))
                fg_x1.append(false_start)
                fg_x2.append(false_end)

            fg_x1 = torch.tensor(fg_x1)
            fg_x2 = torch.tensor(fg_x2)
            return results, fg_x1, fg_x2 - fg_x1

        row_chunks, ys, hs = find_largest_chunk(is_row_with_backgnd)
        col_chunks, xs, ws = find_largest_chunk(is_col_with_backgnd)

        ret_backgnds = []
        for img, row_chunk, col_chunk in zip(x, row_chunks, col_chunks):
            # No background
            if row_chunk == (-1, -1) and col_chunk == (-1, -1):
                ret_backgnds.append(torch.zeros_like(img))
                continue

            # No foreground
            if row_chunk == (0, img.shape[-2]) and col_chunk == (0, img.shape[-1]):
                ret_backgnds.append(img)
                continue

            if row_chunk == (-1, -1):
                chunk = img[:, :, col_chunk[0]:col_chunk[1]]
            elif col_chunk == (-1, -1):
                chunk = img[:, row_chunk[0]:row_chunk[1], :]
            else:
                if row_chunk[1] - row_chunk[0] >= col_chunk[1] - col_chunk[0]:
                    chunk = img[:, row_chunk[0]:row_chunk[1], :]
                else:
                    chunk = img[:, :, col_chunk[0]:col_chunk[1]]

            (img_h, img_w), (h, w) = img.shape[-2:], chunk.shape[-2:]
            ret_b = chunk.repeat(1, int(math.ceil(img_h / h)), int(math.ceil(img_w / w)))
            ret_b = ret_b[:, :img_h, :img_w]
            ret_backgnds.append(ret_b)

        ret_backgnds = torch.stack(ret_backgnds, dim=0)
        return ret_backgnds, (xs, ys, ws, hs)


class FactualMixedRandomTileInpainter(InpaintTemplate):
    '''
    Used in Factual generation with mixed random background.
    It randomly mixes the background within the batch regardless of each class.
    Here mask = 1 means foreground.
    '''
    def __init__(self):
        super().__init__()
        self.cf_tile = TileInpainter(use_bbox_to_mask=True)

    def forward(self, X, mask, y):
        if X.shape[0] == 1 or len(torch.unique(y)) == 1:
            # In edge cases where only 1 example or just 1 class
            # Put 0 in the background
            return X * mask

        # Use rectangular to tile the object to avoid having shape bias!
        bg = self.cf_tile(X, 1 - mask)

        # First, create a y x y array to label which idx isn't the same cls
        tmp = y.reshape(-1, 1).expand(len(y), len(y))
        indicators = (tmp != tmp.T)

        # Then follow https://discuss.pytorch.org/t/efficiently-selecting-a-random-element-from-a-vector/37757/4
        valid_idx = indicators.nonzero()
        unique_rows = valid_idx[:, 0].unique()
        valid_row_idx = [valid_idx[valid_idx[:, 0] == u] for u in unique_rows]

        swap = []
        for v in valid_row_idx:
            choice = torch.multinomial(torch.ones(v.size(0)).float(), 1)
            swap.append(v[choice].squeeze()[1])
        swap = torch.stack(swap)

        return X * mask + bg[swap] * (1 - mask)


class RandomColorWithNoiseInpainter(InpaintTemplate):
    def __init__(self, color_mean=(0.5,), color_std=(0.5,)):
        super(RandomColorWithNoiseInpainter, self).__init__()
        self.color_mean = color_mean
        self.color_std = color_std

    def impute_missing_imgs(self, x, mask):
        background = self.generate_background(x, mask)
        return x * mask + background * (1. - mask)

    def generate_background(self, x, mask):
        random_img = x.new(x.size(0), x.size(1), 1, 1).uniform_().repeat(1, 1, x.size(2), x.size(3))
        random_img += x.new(*x.size()).normal_(0, 0.2)
        random_img.clamp_(0., 1.)

        for c in range(x.size(1)):
            random_img[:, c, :, :].sub_(self.color_mean[c]).div_(self.color_std[c])
        return random_img



