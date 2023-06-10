# import math
# def conv_out_size_same(size, stride):
#   return int(math.ceil(float(size) / float(stride)))
# out=64
# for scaler in range(2, 10, 2):
#   out = conv_out_size_same(out, 2)
#   print(out)

import tensorflow as tf
from tensorflow.keras import layers # pyright: ignore
import math

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

output_height = output_width = 64
gf_dim=64
z_dim=100
c_dim=3
# z = tf.Variable(tf.zeros([None, z_dim]), dtype=tf.float32)
z = tf.zeros([1, z_dim], dtype=tf.float32)
s_h, s_w = output_height, output_width
s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)



model = tf.keras.Sequential()
      
model.add(layers.Dense(gf_dim*8*s_h16*s_w16, use_bias=True, input_shape=[None, z.get_shape()[1]]))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())

model.add(layers.Reshape((s_h16, s_w16, gf_dim * 8)))
assert model.output_shape == (None, s_h16, s_w16, gf_dim * 8)  # Note: None is the batch size

model.add(layers.Conv2DTranspose(gf_dim*4, (5, 5), strides=(2, 2), padding='same', use_bias=True))
assert model.output_shape == (None, s_h8, s_w8, gf_dim*4)
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())

model.add(layers.Conv2DTranspose(gf_dim*2, (5, 5), strides=(2, 2), padding='same', use_bias=True))
assert model.output_shape == (None, s_h4, s_w4, gf_dim*2)
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())

model.add(layers.Conv2DTranspose(gf_dim, (5, 5), strides=(2, 2), padding='same', use_bias=True))
assert model.output_shape == (None, s_h2, s_w2, gf_dim)
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())

model.add(layers.Conv2DTranspose(c_dim, (5, 5), strides=(2, 2), padding='same', use_bias=True, activation='tanh'))
assert model.output_shape == (None, s_h, s_w, c_dim)


x=0










# #region inspect argparse to json error - Namespace is not subscriptable
# import os
# import json
# import numpy as np
# import argparse


# parser = argparse.ArgumentParser(description="Train, Test or Infer DCGAN")

# parser.add_argument("--epoch", type=int, default=25, help="Epoch to train [25]")
# parser.add_argument("--learning-rate", type=float, default=0.0002, help="Learning rate of for adam [0.0002]")
# parser.add_argument("--beta1", type=float, default=0.5, help="Momentum term of adam [0.5]")
# parser.add_argument("--train-size", default=np.inf, help="The size of train images [np.inf]")
# parser.add_argument("--batch-size", type=int, default=64, help="The size of batch images [64]")
# parser.add_argument("--input-height", type=int, default=108, help="The size of image to use (will be center cropped). [108]")
# parser.add_argument("--input-width", type=int, default=None, help="The size of image to use (will be center cropped). If None, same value as input_height [None]")
# parser.add_argument("--output-height", type=int, default=64, help="The size of the output images to produce [64]")
# parser.add_argument("--output-width", type=int, default=None, help="The size of the output images to produce. If None, same value as output_height [None]")
# parser.add_argument("--dataset", type=str, default="celebA", help="The name of dataset [celebA, mnist, lsun]")
# parser.add_argument("--input-fname-pattern", type=str, default="*.jpg", help="Glob pattern of filename of input images [*]")
# parser.add_argument("--data-dir", type=str, default="./data", help="path to datasets [e.g. $HOME/data]")
# parser.add_argument("--out-dir", type=str, default="./out", help="Root directory for outputs [e.g. $HOME/out]")
# parser.add_argument("--out-name", type=str, default="", help="Folder (under out_root_dir) for all outputs. Generated automatically if left blank []")
# parser.add_argument("--checkpoint-dir", type=str, default="checkpoint", help="Folder (under out_root_dir/out_name) to save checkpoints [checkpoint]")
# parser.add_argument("--sample-dir", type=str, default="samples", help="Folder (under out_root_dir/out_name) to save samples [samples]")
# parser.add_argument("--train", type=bool, default=False, help="True for training, False for testing [False]")
# parser.add_argument("--crop", type=bool, default=False, help="True for training, False for testing [False]")
# parser.add_argument("--visualize", type=bool, default=False, help="True for visualizing, False for nothing [False]")
# parser.add_argument("--export", type=bool, default=False, help="True for exporting with new batch size")
# parser.add_argument("--freeze", type=bool, default=False, help="True for exporting with new batch size")
# parser.add_argument("--max-to-keep", type=int, default=1, help="maximum number of checkpoints to keep")
# parser.add_argument("--sample-freq", type=int, default=200, help="sample every this many iterations")
# parser.add_argument("--ckpt-freq", type=int, default=200, help="save checkpoint every this many iterations")
# parser.add_argument("--z-dim", type=int, default=100, help="dimensions of z")
# parser.add_argument("--z-dist", type=str, default="uniform_signed", help="'normal01' or 'uniform_unsigned' or uniform_signed")
# parser.add_argument("--G-img-sum", type=bool, default=False, help="Save generator image summaries in log")
# parser.add_argument("--generate-test-images", type=int, default=100, help="Number of images to generate during test. [100]")


# args = parser.parse_args()
# print(args)
  

# with open(os.path.join(args.out_dir, 'arguments.json'), 'w') as f:
#     arguments_dict = vars(args)
#     json.dump(arguments_dict, f, indent=4, sort_keys=True, ensure_ascii=False)
# #endregion
#
# # region expand path
# import os
# x = "data"
# print(os.path.expanduser(os.path.expandvars(x)))

# #endregion
