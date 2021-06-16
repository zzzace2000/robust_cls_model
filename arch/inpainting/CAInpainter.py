import time

import numpy as np
import tensorflow as tf
import torch
from torch.nn import functional as F


class CAInpainter(object):
    '''
    Contextual Attention GAN inpainter.
    Adapted from https://github.com/JiahuiYu/generative_inpainting
    '''
    def __init__(self, batch_size, checkpoint_dir,
                 pth_mean=(0.5, 0.5, 0.5),
                 pth_std=(0.5, 0.5, 0.5)):
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        tf.compat.v1.disable_eager_execution()

        self.images_ph = tf.compat.v1.placeholder(tf.float32,
                                        shape=[batch_size, 256, 512, 3])
        # with tf.device('/device:GPU:0'):
        # with tf.device('/cpu:0'):
        output = self.build_server_graph(self.images_ph)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        self.output = output

        # load pretrained model
        vars_list = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
        self.assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.train.load_variable(checkpoint_dir, from_name)
            # var_value = tf.contrib.framework.load_variable(checkpoint_dir, from_name)
            self.assign_ops.append(tf.compat.v1.assign(var, var_value))
        print('Model loaded.')

        self.pth_mean = np.ones((1, 3, 1, 1), dtype='float32')
        self.pth_mean[0, :, 0, 0] = np.array(pth_mean)
        self.pth_std = np.ones((1, 3, 1, 1), dtype='float32')
        self.pth_std[0, :, 0, 0] = np.array(pth_std)

        self.upsample = torch.nn.Upsample(size=(256, 256), mode='bilinear')
        # self.downsample = torch.nn.Upsample(size=inpaint_img_size, mode='bilinear')

        # Create a session
        sess_config = tf.compat.v1.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        # sess_config.log_device_placement = True
        self.sess = tf.compat.v1.Session(config=sess_config)

        self.sess.run(self.assign_ops)

    def __call__(self, img, mask):
        return self.impute_missing_imgs(img, mask)

    def impute_missing_imgs(self, img, mask):
        '''
        :param img: 1 x 3 x 224 x 224
        :param mask: 1 x 3 x 224 x 224. Mask
        :return:
        '''
        # If the passed in batch size is smaller than the specified one,
        # Put 0 to make it as a full batch
        orig_batch_size = img.shape[0]
        if orig_batch_size < self.batch_size:
            tmp = img.new_zeros(self.batch_size - orig_batch_size, *img.shape[1:])
            img = torch.cat([img, tmp], dim=0)

            tmp = mask.new_zeros(self.batch_size - orig_batch_size, *mask.shape[1:])
            mask = torch.cat([mask, tmp], dim=0)

        bgd_img = self.generate_background(img, mask)

        result = img * mask + bgd_img * (1. - mask)
        return result[:orig_batch_size]

    def generate_background(self, pytorch_image, pytorch_mask):
        '''
        Use to generate whole blurry images with pytorch normalization.
        '''
        orig_size = pytorch_image.shape[2:]

        mask = pytorch_mask.expand(pytorch_mask.shape[0], 3,
                                   orig_size[0], orig_size[1])
        mask = self.upsample(mask).data.round()
        mask = mask.cpu().numpy()
        # Make it into tensorflow input ordering, then resizing then normalization
        # Do 3 things:
        # - Move from NCHW to NHWC, and from RGB to BGR input
        # - Normalize to 0 - 255 with integer round up
        # - Resize the image size to be 256 x 256

        mask = np.moveaxis(mask, 1, -1)
        mask = (1. - mask) * 255

        image = self.upsample(pytorch_image).data.cpu().numpy()
        image = np.round((image * self.pth_std + self.pth_mean) * 255)
        image = np.moveaxis(image, 1, -1)
        image = image[:, :, :, ::-1]

        input_image = np.concatenate([image, mask], axis=2)

        # DEBUG
        # cv2.imwrite('./test_input.png', input_image[0])

        tf_images = self.sess.run(self.output, {self.images_ph: input_image})

        # it's RGB back. So just change back to pytorch normalization
        pth_img = np.moveaxis(tf_images, 3, 1)
        pth_img = ((pth_img / 255.) - self.pth_mean) / self.pth_std

        pth_img = torch.from_numpy(pth_img)
        pth_img = pth_img.to(pytorch_image.device)
        pth_img = F.interpolate(pth_img, size=orig_size, mode='bilinear').data

        return pth_img

    @classmethod
    def build_server_graph(cls, batch_data, reuse=False, is_training=False):
        # generate mask, 1 represents masked point
        batch_raw, masks_raw = tf.split(batch_data, 2, axis=2)
        masks = tf.cast(masks_raw[:, :, :, 0:1] > 127.5, tf.float32)

        batch_pos = batch_raw / 127.5 - 1.
        batch_incomplete = batch_pos * (1. - masks)
        # inpaint
        x1, x2, flow = cls.build_inpaint_net(
            batch_incomplete, masks, reuse=reuse, training=is_training)
        batch_predict = x2
        # apply mask and reconstruct
        batch_complete = batch_predict*masks + batch_incomplete*(1-masks)
        return batch_complete

    @classmethod
    def build_inpaint_net(cls, x, mask, reuse=False,
                          training=True, padding='SAME',
                          name='inpaint_net'):
        """Inpaint network.

        Args:
            x: incomplete image, [-1, 1]
            mask: mask region {0, 1}
        Returns:
            [-1, 1] as predicted image
        """
        xin = x
        ones_x = tf.ones_like(x)[:, :, :, 0:1]
        x = tf.concat([x, ones_x, ones_x*mask], axis=3)

        # two stage network
        cnum = 32
        with tf.compat.v1.variable_scope(name, reuse=reuse):
            # stage1
            x = cls.gen_conv(x, cnum, 5, 1, name='conv1', padding=padding)
            x = cls.gen_conv(x, 2*cnum, 3, 2, name='conv2_downsample', padding=padding)
            x = cls.gen_conv(x, 2*cnum, 3, 1, name='conv3', padding=padding)
            x = cls.gen_conv(x, 4*cnum, 3, 2, name='conv4_downsample', padding=padding)
            x = cls.gen_conv(x, 4*cnum, 3, 1, name='conv5', padding=padding)
            x = cls.gen_conv(x, 4*cnum, 3, 1, name='conv6', padding=padding)
            mask_s = cls.resize_mask_like(mask, x)
            x = cls.gen_conv(x, 4*cnum, 3, rate=2, name='conv7_atrous', padding=padding)
            x = cls.gen_conv(x, 4*cnum, 3, rate=4, name='conv8_atrous', padding=padding)
            x = cls.gen_conv(x, 4*cnum, 3, rate=8, name='conv9_atrous', padding=padding)
            x = cls.gen_conv(x, 4*cnum, 3, rate=16, name='conv10_atrous', padding=padding)
            x = cls.gen_conv(x, 4*cnum, 3, 1, name='conv11', padding=padding)
            x = cls.gen_conv(x, 4*cnum, 3, 1, name='conv12', padding=padding)
            x = cls.gen_deconv(x, 2*cnum, name='conv13_upsample', padding=padding)
            x = cls.gen_conv(x, 2*cnum, 3, 1, name='conv14', padding=padding)
            x = cls.gen_deconv(x, cnum, name='conv15_upsample', padding=padding)
            x = cls.gen_conv(x, cnum//2, 3, 1, name='conv16', padding=padding)
            x = cls.gen_conv(x, 3, 3, 1, activation=None, name='conv17', padding=padding)
            x = tf.clip_by_value(x, -1., 1.)
            x_stage1 = x
            # return x_stage1, None, None

            # stage2, paste result as input
            # x = tf.stop_gradient(x)
            x = x*mask + xin*(1.-mask)
            x.set_shape(xin.get_shape().as_list())
            # conv branch
            xnow = tf.concat([x, ones_x, ones_x*mask], axis=3)
            x = cls.gen_conv(xnow, cnum, 5, 1, name='xconv1', padding=padding)
            x = cls.gen_conv(x, cnum, 3, 2, name='xconv2_downsample', padding=padding)
            x = cls.gen_conv(x, 2*cnum, 3, 1, name='xconv3', padding=padding)
            x = cls.gen_conv(x, 2*cnum, 3, 2, name='xconv4_downsample', padding=padding)
            x = cls.gen_conv(x, 4*cnum, 3, 1, name='xconv5', padding=padding)
            x = cls.gen_conv(x, 4*cnum, 3, 1, name='xconv6', padding=padding)
            x = cls.gen_conv(x, 4*cnum, 3, rate=2, name='xconv7_atrous', padding=padding)
            x = cls.gen_conv(x, 4*cnum, 3, rate=4, name='xconv8_atrous', padding=padding)
            x = cls.gen_conv(x, 4*cnum, 3, rate=8, name='xconv9_atrous', padding=padding)
            x = cls.gen_conv(x, 4*cnum, 3, rate=16, name='xconv10_atrous', padding=padding)
            x_hallu = x
            # attention branch
            x = cls.gen_conv(xnow, cnum, 5, 1, name='pmconv1', padding=padding)
            x = cls.gen_conv(x, cnum, 3, 2, name='pmconv2_downsample', padding=padding)
            x = cls.gen_conv(x, 2*cnum, 3, 1, name='pmconv3', padding=padding)
            x = cls.gen_conv(x, 4*cnum, 3, 2, name='pmconv4_downsample', padding=padding)
            x = cls.gen_conv(x, 4*cnum, 3, 1, name='pmconv5', padding=padding)
            x = cls.gen_conv(x, 4*cnum, 3, 1, name='pmconv6', padding=padding,
                              activation=tf.nn.relu)
            x, offset_flow = cls.contextual_attention(x, x, mask_s, 3, 1, rate=2)
            x = cls.gen_conv(x, 4*cnum, 3, 1, name='pmconv9', padding=padding)
            x = cls.gen_conv(x, 4*cnum, 3, 1, name='pmconv10', padding=padding)
            pm = x
            # pm = tf.zeros([1, 64, 64, 128])
            # x_hallu, pm: 1x64x64x128
            x = tf.concat([x_hallu, pm], axis=3)

            x = cls.gen_conv(x, 4*cnum, 3, 1, name='allconv11', padding=padding)
            x = cls.gen_conv(x, 4*cnum, 3, 1, name='allconv12', padding=padding)
            x = cls.gen_deconv(x, 2*cnum, name='allconv13_upsample', padding=padding)
            x = cls.gen_conv(x, 2*cnum, 3, 1, name='allconv14', padding=padding)
            x = cls.gen_deconv(x, cnum, name='allconv15_upsample', padding=padding)
            x = cls.gen_conv(x, cnum//2, 3, 1, name='allconv16', padding=padding)
            x = cls.gen_conv(x, 3, 3, 1, activation=None, name='allconv17', padding=padding)
            x_stage2 = tf.clip_by_value(x, -1., 1.)
        return x_stage1, x_stage2, offset_flow

    @classmethod
    def gen_conv(cls, x, cnum, ksize, stride=1, rate=1, name='conv',
                 padding='SAME', activation=tf.nn.elu):
        """Define conv for generator.

        Args:
            x: Input.
            cnum: Channel number.
            ksize: Kernel size.
            Stride: Convolution stride.
            Rate: Rate for or dilated conv.
            name: Name of layers.
            padding: Default to SYMMETRIC.
            activation: Activation function after convolution.
            training: If current graph is for training or inference, used for bn.

        Returns:
            tf.Tensor: output

        """
        assert padding in ['SYMMETRIC', 'SAME', 'REFELECT']
        if padding == 'SYMMETRIC' or padding == 'REFELECT':
            p = int(rate * (ksize - 1) / 2)
            x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], mode=padding)
            padding = 'VALID'
        x = tf.compat.v1.layers.conv2d(
            x, cnum, ksize, stride, dilation_rate=rate,
            activation=activation, padding=padding, name=name)
        return x

    @classmethod
    def gen_deconv(cls, x, cnum, name='upsample', padding='SAME'):
        """Define deconv for generator.
        The deconv is defined to be a x2 resize_nearest_neighbor operation with
        additional gen_conv operation.

        Args:
            x: Input.
            cnum: Channel number.
            name: Name of layers.
            training: If current graph is for training or inference, used for bn.

        Returns:
            tf.Tensor: output

        """
        with tf.compat.v1.variable_scope(name):
            x = cls.resize(x, func=tf.compat.v1.image.resize_nearest_neighbor)
            x = cls.gen_conv(
                x, cnum, 3, 1, name=name + '_conv', padding=padding)
        return x

    @classmethod
    def contextual_attention(cls, f, b, mask=None, ksize=3, stride=1, rate=1,
                             fuse_k=3, softmax_scale=10., training=True, fuse=True):
        """ Contextual attention layer implementation.

        Contextual attention is first introduced in publication:
            Generative Image Inpainting with Contextual Attention, Yu et al.

        Args:
            x: Input feature to match (foreground).
            t: Input feature for match (background).
            mask: Input mask for t, indicating patches not available.
            ksize: Kernel size for contextual attention.
            stride: Stride for extracting patches from t.
            rate: Dilation for matching.
            softmax_scale: Scaled softmax for attention.
            training: Indicating if current graph is training or inference.

        Returns:
            tf.Tensor: output

        """
        # get shapes
        raw_fs = tf.shape(f)
        raw_int_fs = f.get_shape().as_list()
        raw_int_bs = b.get_shape().as_list()
        # extract patches from background with stride and rate
        kernel = 2 * rate
        raw_w = tf.compat.v1.extract_image_patches(
            b, [1, kernel, kernel, 1], [1, rate * stride, rate * stride, 1], [1, 1, 1, 1], padding='SAME')
        raw_w = tf.reshape(raw_w, [raw_int_bs[0], -1, kernel, kernel, raw_int_bs[3]])
        raw_w = tf.transpose(raw_w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
        # downscaling foreground option: downscaling both foreground and
        # background for matching and use original background for reconstruction.
        f = cls.resize(f, scale=1. / rate, func=tf.compat.v1.image.resize_nearest_neighbor)
        b = cls.resize(b, to_shape=[int(raw_int_bs[1] / rate), int(raw_int_bs[2] / rate)],
                   func=tf.compat.v1.image.resize_nearest_neighbor)  # https://github.com/tensorflow/tensorflow/issues/11651

        fs = tf.shape(f)
        int_fs = f.get_shape().as_list()
        f_groups = tf.split(f, int_fs[0], axis=0)
        # from t(H*W*C) to w(b*k*k*c*h*w)
        bs = tf.shape(b)
        int_bs = b.get_shape().as_list()
        w = tf.compat.v1.extract_image_patches(
            b, [1, ksize, ksize, 1], [1, stride, stride, 1], [1, 1, 1, 1], padding='SAME')
        w = tf.reshape(w, [int_fs[0], -1, ksize, ksize, int_fs[3]])
        w = tf.transpose(w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw

        # process mask
        if mask is None:
            mask = tf.zeros([1, bs[1], bs[2], 1])
        else:
            mask = cls.resize(mask, scale=1. / rate, func=tf.compat.v1.image.resize_nearest_neighbor)

        m_patches = tf.compat.v1.extract_image_patches(
            mask, [1, ksize, ksize, 1], [1, stride, stride, 1], [1, 1, 1, 1], padding='SAME')
        m_patches = tf.reshape(m_patches, [m_patches.get_shape().as_list()[0], -1, ksize, ksize, 1])
        m_patches = tf.transpose(m_patches, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
        m_patches = tf.cast(tf.equal(tf.reduce_mean(m_patches, axis=[1, 2, 3], keepdims=True), 0.), tf.float32)

        w_norm = w / tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(w), axis=[1, 2, 3], keepdims=True)), 1e-4)
        w_norm_groups = tf.split(w_norm, int_bs[0], axis=0)
        raw_w_groups = tf.split(raw_w, int_bs[0], axis=0)
        y = []
        offsets = []
        k = fuse_k
        scale = softmax_scale
        fuse_weight = tf.reshape(tf.eye(k), [k, k, 1, 1])

        for i, (xi, wi_normed, raw_wi) in enumerate(zip(f_groups, w_norm_groups, raw_w_groups)):
            mm = m_patches[i]

            # conv for compare
            wi_normed = wi_normed[0]
            yi = tf.nn.conv2d(xi, wi_normed, strides=[1, 1, 1, 1], padding="SAME")

            # conv implementation for fuse scores to encourage large patches
            if fuse:
                yi = tf.reshape(yi, [1, fs[1] * fs[2], bs[1] * bs[2], 1])
                yi = tf.nn.conv2d(yi, fuse_weight, strides=[1, 1, 1, 1], padding='SAME')
                yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1], bs[2]])
                yi = tf.transpose(yi, [0, 2, 1, 4, 3])
                yi = tf.reshape(yi, [1, fs[1] * fs[2], bs[1] * bs[2], 1])
                yi = tf.nn.conv2d(yi, fuse_weight, strides=[1, 1, 1, 1], padding='SAME')
                yi = tf.reshape(yi, [1, fs[2], fs[1], bs[2], bs[1]])
                yi = tf.transpose(yi, [0, 2, 1, 4, 3])
            yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1] * bs[2]])

            # softmax to match
            yi *= mm  # mask
            yi = tf.nn.softmax(yi * scale, 3)
            yi *= mm  # mask

            offset = tf.argmax(yi, axis=3, output_type=tf.int32)
            offset = tf.stack([offset // fs[2], offset % fs[2]], axis=-1)
            # deconv for patch pasting
            # 3.1 paste center
            wi_center = raw_wi[0]
            yi = tf.nn.conv2d_transpose(yi, wi_center, tf.concat([[1], raw_fs[1:]], axis=0),
                                        strides=[1, rate, rate, 1]) / 4.
            y.append(yi)
            offsets.append(offset)
        y = tf.concat(y, axis=0)
        y.set_shape(raw_int_fs)
        offsets = tf.concat(offsets, axis=0)
        offsets.set_shape(int_bs[:3] + [2])
        # case1: visualize optical flow: minus current position
        h_add = tf.tile(tf.reshape(tf.range(bs[1]), [1, bs[1], 1, 1]), [bs[0], 1, bs[2], 1])
        w_add = tf.tile(tf.reshape(tf.range(bs[2]), [1, 1, bs[2], 1]), [bs[0], bs[1], 1, 1])
        offsets = offsets - tf.concat([h_add, w_add], axis=3)
        # to flow image
        flow = cls.flow_to_image_tf(offsets)
        # # case2: visualize which pixels are attended
        # flow = highlight_flow_tf(offsets * tf.cast(mask, tf.int32))
        if rate != 1:
            flow = cls.resize(flow, scale=rate, func=tf.compat.v1.image.resize_nearest_neighbor)
        return y, flow

    @classmethod
    def resize_mask_like(cls, mask, x):
        """Resize mask like shape of x.

        Args:
            mask: Original mask.
            x: To shape of x.

        Returns:
            tf.Tensor: resized mask

        """
        mask_resize = cls.resize(
            mask, to_shape=x.get_shape().as_list()[1:3],
            func=tf.compat.v1.image.resize_nearest_neighbor)
        return mask_resize

    @classmethod
    def flow_to_image_tf(cls, flow, name='flow_to_image'):
        """Tensorflow ops for computing flow to image.
        """
        with tf.compat.v1.variable_scope(name), tf.device('/cpu:0'):
            img = tf.compat.v1.py_func(cls.flow_to_image, [flow], tf.float32, stateful=False)
            img.set_shape(flow.get_shape().as_list()[0:-1] + [3])
            img = img / 127.5 - 1.
            return img

    @classmethod
    def flow_to_image(cls, flow):
        """Transfer flow map to image.
        Part of code forked from flownet.
        """
        out = []
        maxu = -999.
        maxv = -999.
        minu = 999.
        minv = 999.
        maxrad = -1
        for i in range(flow.shape[0]):
            u = flow[i, :, :, 0]
            v = flow[i, :, :, 1]
            idxunknow = (abs(u) > 1e7) | (abs(v) > 1e7)
            u[idxunknow] = 0
            v[idxunknow] = 0
            maxu = max(maxu, np.max(u))
            minu = min(minu, np.min(u))
            maxv = max(maxv, np.max(v))
            minv = min(minv, np.min(v))
            rad = np.sqrt(u ** 2 + v ** 2)
            maxrad = max(maxrad, np.max(rad))
            u = u / (maxrad + np.finfo(float).eps)
            v = v / (maxrad + np.finfo(float).eps)
            img = cls.compute_color(u, v)
            out.append(img)
        return np.float32(np.uint8(out))

    @classmethod
    def compute_color(cls, u, v):
        h, w = u.shape
        img = np.zeros([h, w, 3])
        nanIdx = np.isnan(u) | np.isnan(v)
        u[nanIdx] = 0
        v[nanIdx] = 0
        # colorwheel = COLORWHEEL
        colorwheel = cls.make_color_wheel()
        ncols = np.size(colorwheel, 0)
        rad = np.sqrt(u ** 2 + v ** 2)
        a = np.arctan2(-v, -u) / np.pi
        fk = (a + 1) / 2 * (ncols - 1) + 1
        k0 = np.floor(fk).astype(int)
        k1 = k0 + 1
        k1[k1 == ncols + 1] = 1
        f = fk - k0
        for i in range(np.size(colorwheel, 1)):
            tmp = colorwheel[:, i]
            col0 = tmp[k0 - 1] / 255
            col1 = tmp[k1 - 1] / 255
            col = (1 - f) * col0 + f * col1
            idx = rad <= 1
            col[idx] = 1 - rad[idx] * (1 - col[idx])
            notidx = np.logical_not(idx)
            col[notidx] *= 0.75
            img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))
        return img

    @classmethod
    def make_color_wheel(cls):
        RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
        ncols = RY + YG + GC + CB + BM + MR
        colorwheel = np.zeros([ncols, 3])
        col = 0
        # RY
        colorwheel[0:RY, 0] = 255
        colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
        col += RY
        # YG
        colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
        colorwheel[col:col + YG, 1] = 255
        col += YG
        # GC
        colorwheel[col:col + GC, 1] = 255
        colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
        col += GC
        # CB
        colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
        colorwheel[col:col + CB, 2] = 255
        col += CB
        # BM
        colorwheel[col:col + BM, 2] = 255
        colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
        col += + BM
        # MR
        colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
        colorwheel[col:col + MR, 0] = 255
        return colorwheel

    @classmethod
    def resize(cls, x, scale=2., to_shape=None, align_corners=True, dynamic=False,
               func=tf.compat.v1.image.resize_bilinear, name='resize'):
        if dynamic:
            xs = tf.cast(tf.shape(x), tf.float32)
            new_xs = [tf.cast(xs[1] * scale, tf.int32),
                      tf.cast(xs[2] * scale, tf.int32)]
        else:
            xs = x.get_shape().as_list()
            new_xs = [int(xs[1] * scale), int(xs[2] * scale)]
        with tf.compat.v1.variable_scope(name):
            if to_shape is None:
                x = func(x, new_xs, align_corners=align_corners)
            else:
                x = func(x, [to_shape[0], to_shape[1]],
                         align_corners=align_corners)
        return x

    def time_impute_missing_imgs(self, pytorch_image, pytorch_mask):
        start_time = time.time()
        result = self.impute_missing_imgs(pytorch_image, pytorch_mask)
        print('Total time:', time.time() - start_time)
        return result

    def reset(self):
        pass

    def eval(self):
        pass

    def cuda(self):
        self.upsample.cuda()
        self.downsample.cuda()

    def __del__(self):
        self.sess.close()
