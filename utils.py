"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import logging
import random
import itertools
import numpy as np
import os
from time import gmtime, strftime
from six.moves import xrange
from PIL import Image
from glob import glob


def expand_path(path):
    return os.path.expanduser(os.path.expandvars(path))


def save_images(images, size, image_path):
    return im_save(inverse_transform(images), size, image_path)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if images.shape[3] in (3, 4):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                         'must have dimensions: HxW or HxWx3 or HxWx4')


def im_save(images, size, path):
    image = np.squeeze(merge(images, size))
    if image.dtype == np.float64:
        return Image.fromarray((image * 255).astype(np.uint8)).save(path) # TODO :: check un-normalization - do I need to dis-zero-center images as well
    return Image.fromarray(image).save(path)


def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    im = Image.fromarray(x[j:j + crop_h, i:i + crop_w])
    return np.array(im.resize([resize_h, resize_w]), dtype=np.float32)


def transform(image, input_height, input_width,
              resize_height=64, resize_width=64, crop=True):
    if crop:
        final_return = center_crop(
            image, input_height, input_width,
            resize_height, resize_width)
    else:
        image = Image.fromarray(image)
        final_return = np.array(image.resize([input_height, input_width]), dtype=np.float32)
    # if final_return.shape[-1] != 3:
    #     final_return = np.reshape(final_return, [final_return.shape[0], final_return.shape[1], 1], )
    final_return /= 255 # normalize 0-1
    # # zero - center images (using image-net MEAN and STD values - actual zero centring would require to use MEAN and STD of current dataset)
    # final_return[:, :, 0] -= 0.485
    # final_return[:, :, 0] /= 0.299
    # final_return[:, :, 1] -= 0.456
    # final_return[:, :, 1] /= 0.224
    # final_return[:, :, 2] -= 0.406
    # final_return[:, :, 2] /= 0.225
    return final_return


def make_gif(images, fname, duration=2, true_image=False):
    import moviepy.editor as mpy

    def make_frame(t):
        try:
            x = images[int(len(images) / duration * t)]
        except:
            x = images[-1]

        if true_image:
            return x.astype(np.uint8)
        else:
            """
      eager tensor has not attribute as type, enable numpy
      from tensorflow.python.ops.numpy_ops import np_config
        np_config.enable_numpy_behavior()
      """
            from tensorflow.python.ops.numpy_ops import np_config
            np_config.enable_numpy_behavior()
            return (x * 255).astype(np.uint8) # TODO :: check un-normalization - do I need to dis-zero-center images as well

    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.write_gif(fname, fps=len(images) / duration)


def visualize(generator_model, config, option, sample_dir):
    image_frame_dim = int(math.ceil(config.batch_size ** .5))
    if option == 0:
        z_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, config.z_dim))
        samples = generator_model(z_sample)
        save_images(samples, [image_frame_dim, image_frame_dim],
                    os.path.join(sample_dir, 'test_%s.png' % strftime("%Y%m%d%H%M%S", gmtime())))
    elif option == 1:
        values = np.arange(0, 1, 1. / config.batch_size)
        for idx in xrange(config.z_dim):
            print(" [*] %d" % idx)
            z_sample = np.random.uniform(-1, 1, size=(config.batch_size, config.z_dim))
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]

            samples = generator_model(z_sample)
            save_images(samples, [image_frame_dim, image_frame_dim],
                        os.path.join(sample_dir, 'test_arange_%s.png' % idx))
    elif option == 2:
        values = np.arange(0, 1, 1. / config.batch_size)
        for idx in [random.randint(0, config.z_dim - 1) for _ in xrange(config.z_dim)]:
            print(" [*] %d" % idx)
            z = np.random.uniform(-0.2, 0.2, size=config.z_dim)
            z_sample = np.tile(z, (config.batch_size, 1))
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]
            samples = generator_model(z_sample)
            try:
                make_gif(samples, './samples/test_gif_%s.gif' % idx)
            except:
                save_images(samples, [image_frame_dim, image_frame_dim],
                            os.path.join(sample_dir, 'test_%s.png' % strftime("%Y%m%d%H%M%S", gmtime())))
    elif option == 3:
        values = np.arange(0, 1, 1. / config.batch_size)
        for idx in xrange(config.z_dim):
            print(" [*] %d" % idx)
            z_sample = np.zeros([config.batch_size, config.z_dim])
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]

            samples = generator_model(z_sample)
            make_gif(samples, os.path.join(sample_dir, 'test_gif_%s.gif' % idx))
    elif option == 4:
        image_set = []
        values = np.arange(0, 1, 1. / config.batch_size)

        for idx in xrange(config.z_dim):
            print(" [*] %d" % idx)
            # z_sample = np.zeros([config.batch_size, config.z_dim])
            # https://github.com/carpedm20/DCGAN-tensorflow/issues/204#issuecomment-339016947
            z_sample = np.random.uniform(-1, 1, size=(config.batch_size, config.z_dim))
            for kdx, z in enumerate(z_sample): z[idx] = values[kdx]

            image_set.append(generator_model(z_sample, training=False))
            if image_set[0].shape[-1] != 3:
                logging.info("Cannot generate gif for images with single channel")
                return
            make_gif(image_set[-1], os.path.join(sample_dir, 'test_gif_%s.gif' % idx))

        new_image_set = [merge(np.array([images[idx] for images in image_set]), [10, 10])
                         for idx in itertools.chain(range(config.batch_size), range(config.batch_size-1, -1, -1))]
        _ = [os.remove(gifs) for gifs in glob(os.path.join(sample_dir, '*.gif'))]
        make_gif(new_image_set, os.path.join(sample_dir, 'test_gif_merged.gif'), duration=8)
