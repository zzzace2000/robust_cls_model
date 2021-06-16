# -*- coding: utf-8 -*-
"""
Some utility functions for visualisation, not documented properly
"""

from skimage import color
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pylab
from torchvision.utils import make_grid
import torch
import matplotlib.patches as patches
import math


def plot_results(x_test, x_test_im, sensMap, predDiff, tarFunc, classnames, testIdx, save_path):
    '''
    Plot the results of the relevance estimation
    '''
    imsize = x_test.shape

    tarIdx = np.argmax(tarFunc(x_test)[-1])
    tarClass = classnames[tarIdx]
    # tarIdx = 287

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(x_test_im, interpolation='nearest')
    plt.title('original')
    frame = pylab.gca()
    frame.axes.get_xaxis().set_ticks([])
    frame.axes.get_yaxis().set_ticks([])
    plt.subplot(2, 2, 2)
    plt.imshow(sensMap, cmap=cm.Greys_r, interpolation='nearest')
    plt.title('sensitivity map')
    frame = pylab.gca()
    frame.axes.get_xaxis().set_ticks([])
    frame.axes.get_yaxis().set_ticks([])
    plt.subplot(2, 2, 3)
    p = predDiff.reshape((imsize[1], imsize[2], -1))[:, :, tarIdx]
    plt.imshow(p, cmap=cm.seismic, vmin=-np.max(np.abs(p)), vmax=np.max(np.abs(p)), interpolation='nearest')
    plt.colorbar()
    # plt.imshow(np.abs(p), cmap=cm.Greys_r)
    plt.title('weight of evidence')
    frame = pylab.gca()
    frame.axes.get_xaxis().set_ticks([])
    frame.axes.get_yaxis().set_ticks([])
    plt.subplot(2, 2, 4)
    plt.title('class: {}'.format(tarClass))
    p = get_overlayed_image(x_test_im, p)
    # p = predDiff[0,:,:,np.argmax(netPred(net, x_test)[0]),1].reshape((224,224))
    plt.imshow(p, cmap=cm.seismic, vmin=-np.max(np.abs(p)), vmax=np.max(np.abs(p)), interpolation='nearest')
    # plt.title('class entropy')
    frame = pylab.gca()
    frame.axes.get_xaxis().set_ticks([])
    frame.axes.get_yaxis().set_ticks([])

    fig = plt.gcf()
    fig.set_size_inches(np.array([12, 12]), forward=True)
    plt.tight_layout()
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def pytorch_to_np(pytorch_image):
    if pytorch_image.ndim == 4 and pytorch_image.shape[1] == 1:
        pytorch_image = pytorch_image.repeat(1, 3, 1, 1)
    if pytorch_image.ndim == 3 and pytorch_image.shape[0] == 1:
        pytorch_image = pytorch_image.repeat(3, 1, 1)

    return pytorch_image.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()


def plot_pytorch_img(pytorch_img, ax=None, cmap=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    return ax.imshow(pytorch_to_np(pytorch_img), cmap=cmap, interpolation='nearest', **kwargs)


def plot_rectangle(x, y, w, h, ax, color='red'):
    from matplotlib import patches
    x = x.item() if isinstance(x, torch.Tensor) else x
    y = y.item() if isinstance(y, torch.Tensor) else y
    w = w.item() if isinstance(w, torch.Tensor) else w
    h = h.item() if isinstance(h, torch.Tensor) else h

    if x == -1:
        return

    ax.add_patch(
        patches.Rectangle(
            xy=(x, y), width=w, height=h,
            color=color, fill=False  # remove background
        )
    )


def _preprocess_img_to_pytorch(img):
    if type(img) == np.ndarray:
        img = torch.FloatTensor(img)
    if img.ndimension() != 3:
        raise Exception('The input dimension of image is not 3 but %d' % img.ndimension())
    if img.shape[0] == 1:
        img = img.expand(3, img.shape[1], img.shape[2])
    return img


def plot_orig_and_overlay_img(orig_img, overlayed_img, file_name=None, bbox_coord=None, gt_class_name='',
                              pred_class_name='', cmap=cm.seismic, clim=None, visualize=False):
    '''
    :param orig_img: PyTorch 3d array [channel, width, height]
    :param overlayed_img: PyTorch 3d array [channel, width, height]
    :param visualize: Default True.
    :return:
    '''
    orig_img = _preprocess_img_to_pytorch(orig_img)
    overlayed_img = _preprocess_img_to_pytorch(overlayed_img)

    if type(overlayed_img) == np.ndarray:
        overlayed_img = torch.from_numpy(overlayed_img)

    plt.close()
    fig = plt.figure()

    # Plot original image
    ax1 = fig.add_subplot(121)
    im1 = plot_pytorch_img(orig_img, ax1, cmap)
    fig.colorbar(im1, ax=ax1)
    ## Plot the bounding box
    ax1.add_patch(
        patches.Rectangle(
            bbox_coord[0, :], bbox_coord[1, 0] - bbox_coord[0, 0], bbox_coord[1, 1] - bbox_coord[0, 1],
            color='red', fill=False  # remove background
        )
    )

    # Plot the overlayed image
    ax2 = fig.add_subplot(122)
    if clim is not None:
        im2 = plot_pytorch_img(overlayed_img, ax2, cmap=cm.seismic, vmin=clim[0], vmax=clim[1])
        fig.colorbar(im2, ax=ax2, cmap=cm.seismic, fraction=0.046, pad=0.04)
    else:
        im2 = plot_pytorch_img(overlayed_img, ax2, cmap=cm.seismic)

    title = gt_class_name
    if gt_class_name != pred_class_name:
        title = '%s\n%s' % (gt_class_name, pred_class_name)

    plt.title(title)
    ax1.axis("off")
    ax2.axis("off")
    plt.subplots_adjust(left=0.075, bottom=0.2, right=0.9)

    if visualize:
        plt.show()
    else:
        plt.savefig(file_name, dpi=300)
    plt.close()


def get_overlayed_images(orig_imgs, color_vecs, cmap=cm.seismic):
    '''
    :return: color_overlay_img: the image overlayed with noise color
    '''
    assert orig_imgs.ndim == 4 and color_vecs.ndim == 4

    result, clims = [], []
    for orig_img, color_vec in zip(orig_imgs, color_vecs):
        orig_img = orig_img.cpu().detach().numpy()
        color_vec = color_vec.cpu().detach().numpy()
        overlayed_img, clim = overlay(orig_img, color_vec[0], cmap=cmap)
        result.append(torch.from_numpy(overlayed_img))
        clims.append(clim)

    result = torch.stack(result)
    return result, clims


def overlay(x, c, gray_factor_bg=0.3, alpha=0.8, cmap=cm.seismic):
    '''
    For an image x and a relevance vector c, overlay the image with the
    relevance vector to visualise the influence of the image pixels.
    '''
    assert np.ndim(c) <= 2, 'dimension of c is:' + str(np.ndim(c))
    imDim = x.shape[0]

    if np.ndim(c) == 1:
        c = c.reshape((imDim, imDim))

    # this happens with the MNIST Data
    if np.ndim(x) == 2:
        x = 1 - np.dstack((x, x, x)) * gray_factor_bg  # make it a bit grayish

    elif np.ndim(x) == 3:  # this is what happens with cifar data
        x = np.transpose(x, (1, 2, 0))
        x = color.rgb2gray(x)
        x = 1 - (1 - x) * 0.3
        x = np.dstack((x, x, x))

    # Construct a colour image to superimpose
    vlimit = abs(c.min()) if abs(c.min()) > abs(c.max()) else abs(c.max())

    im = plt.imshow(c, cmap=cmap, interpolation='nearest', vmin=-vlimit, vmax=vlimit)
    color_mask = im.to_rgba(c)[:, :, [0, 1, 2]]
    clim = im.properties()['clim']

    # Convert the input image and color mask to Hue Saturation Value (HSV) colorspace
    img_hsv = color.rgb2hsv(x)
    color_mask_hsv = color.rgb2hsv(color_mask)

    # Replace the hue and saturation of the original image
    # with that of the color mask
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)
    img_masked = np.transpose(img_masked, (2, 0, 1))

    plt.close()
    return img_masked, clim


# Visualize and save
def plot_pytorch_imgs(imgs_list, filename='', ncols=4, dpi=300, ax=None, clim=None):
    grid = make_grid(imgs_list, nrow=ncols)
    if ax is None:
        fig, ax = plt.subplots()

    im = plot_pytorch_img(grid, ax, clim=clim)

    if filename != '':
        plt.savefig(filename, dpi=dpi)

    return ax


def plot_imgs_with_bboxes2(samples, ncols=8):
    imgs = samples['imgs']
    xs = samples['xs']
    ys = samples['ys']
    ws = samples['ws']
    hs = samples['hs']

    if imgs.ndim == 3: # only 1 img
        return plot_img_with_bbox(
            imgs, xs, ys, ws, hs)

    num_imgs = len(imgs)
    nrows = int(math.ceil(num_imgs / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5))

    if xs is None:
        for idx, img in enumerate(imgs):
            plot_pytorch_img(img, axes.flat[idx])
    else:
        for idx, (img, x, y, w, h) in enumerate(zip(imgs, xs, ys, ws, hs)):
            plot_img_with_bbox(
                img, x, y, w, h,
                fig=fig, ax=axes.flat[idx])
    return fig, axes


def plot_imgs_with_bboxes(imgs, xs=None, ys=None, ws=None, hs=None, ncols=8):
    num_imgs = len(imgs)
    nrows = int(math.ceil(num_imgs / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5))

    if xs is None:
        for idx, img in enumerate(imgs):
            plot_pytorch_img(img, axes.flat[idx])
    else:
        for idx, (img, x, y, w, h) in enumerate(zip(imgs, xs, ys, ws, hs)):
            plot_img_with_bbox(
                img, x, y, w, h,
                fig=fig,
                ax=axes.flat[idx])

    return fig, axes


def plot_img_with_bbox(img, xs, ys, ws, hs, fig=None, ax=None):
    if fig is None:
        fig, ax = plt.subplots()
    im = plot_pytorch_img(img, ax)

    if torch.is_tensor(xs) and xs.ndim > 0:
        for x, y, w, h in zip(xs, ys, ws, hs):
            plot_rectangle(x, y, w, h, ax=ax)
    else:
        plot_rectangle(xs, ys, ws, hs, ax=ax)

    return fig, ax
