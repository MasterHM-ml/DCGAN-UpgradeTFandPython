import tensorflow as tf
print(tf.random.uniform([2, 2], minval=-1, maxval=1))


# from math import floor
# list_ = list(range(0,1000))

# for batch_index in range(floor(int(1000 / 4))):
#     print(list_[batch_index * 4:(batch_index + 1) * 4])



# import tensorflow as tf

# print(tf.random.normal([2, 2]))



# # import re
# # import os
# # from glob import glob

# # [os.mkdir("checkpoint-%d.data-dir" % indexer) for indexer in range(100)]
# # [os.rmdir("checkpoint-%d.data-dir" % indexer) for indexer in range(100)]
# # x = glob("checkpoint-*")
# # # x.sort(key=lambda x: int(re.search(r'\d+', x).group()))
# # x.sort(key=lambda x: int(x[11:].split(".")[0]))

# # print(x)







# # import os
# # import json

# # os.makedirs("./out/cifar10_2023-06-13/checkpoints/", exist_ok=True)
# # with open(os.path.join("./out/cifar10_2023-06-13/checkpoints/", "image_io.json"), "w") as ff:
# #       json.dump({
# #         "input_height_width": 32,
# #         "output_height_width": 32,
# #       }, ff)

# # with open("./out/cifar10_2023-06-13/checkpoints/image_io.json", "r") as ff:
# #     ff = json.load(ff)
# #     input_height_width = x = ff["input_height_width"]
# #     output_height_width= x = ff["output_height_width"]
# # print(input_height_width, output_height_width)


# # h=w=50
# # if h == w < 32:
# #     raise ValueError("input_height and input_width must be at least 32, got %d height and %d width"\
# #                      % (h, w))
# # h=w=130
# # if h == 32:
# #     pass
# # elif int(h/32) in list(range(2,100,2)):

# #     print(h%32)
# # else:
# #     raise "input_height and input_width must be either 32 or even divisible of 32 e.g. 64, 128, 256, 512 etc."

# # import numpy as np
# # import tensorflow as tf
# # from tqdm import trange
# from PIL import Image
# # (train_images, _), (_, _) = getattr(tf.keras.datasets, "mnist").load_data()
# # # for indexer in trange(train_images.shape[0]):
# # #     train_images[indexer] = np.array(Image.fromarray(train_images[indexer]).resize((32,32)))
# # train_images = tf.pad(train_images, [[0,0], [2,2], [2,2]]).numpy()


# # print(type(train_images))
# print(Image.open("/home/master/hm.jpg").size == (407, 914))



















# # # import time
# # # import datetime

# # # print(str(datetime.date.today()))
# # # print(str(time.asctime()))
# # # print(time.strftime("%Y-%m-%d %H:%M:%S"))


# # # import os
# # # import subprocess
# # # os.mkdir("test_dirs")
# # # for i in range(1000):
# # #     os.mkdir("test_dirs/test_%i"%i)
# # # print(sorted(os.listdir("test_dirs")))
# # # subprocess.call("rm -rf test_dirs", shell=True)


# # # import tensorflow as tf

# # # batch_size=64
# # # (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
# # # train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
# # # train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
# # # buffer_size = train_images.shape[0]
# # # train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(buffer_size).batch(batch_size)

# # # import numpy as np
# # # import PIL
# # # from PIL import Image
# # # import os
# # # from glob import glob
# # # import numpy as np
# # # import tensorflow as tf
# # # # from utils import transform
# # # def center_crop(x, crop_h, crop_w,
# # #                 resize_h=64, resize_w=64):
# # #   if crop_w is None:
# # #     crop_w = crop_h
# # #   h, w = x.shape[:2]
# # #   j = int(round((h - crop_h)/2.))
# # #   i = int(round((w - crop_w)/2.))
# # #   im = Image.fromarray(x[j:j+crop_h, i:i+crop_w])
# # #   return np.array(im.resize([resize_h, resize_w]), dtype=np.float32)/127.5 - 1.0

# # # def transform(image, input_height, input_width, 
# # #               resize_height=64, resize_width=64, crop=True):
# # #   if crop:
# # #     final_return = center_crop(
# # #       image, input_height, input_width, 
# # #       resize_height, resize_width)
# # #   else:
# # #     image = Image.fromarray(image)
# # #     final_return = np.array(image.resize([input_height, input_width]), dtype=np.float32)/127.5 - 1.0
# # #   if final_return.shape[-1] !=3:
# # #     final_return = np.reshape(final_return, [final_return.shape[0], final_return.shape[1],1], )
# # #   return final_return
  
# # # data_path = os.path.join("/media/master/Memories/HM", "*jpg")
# # #     #   logging.info("loading custom data from %s." % data_path)
# # #     #   logging.info("Please make sure all images are either RGB (3 channels) or grayscale (1 channels). Got argument 'c_dim'=%d" % self.c_dim)
# # # batch_size=64
# # # data_X = glob(data_path)
# # # if len(data_X) == 0: raise Exception("[!] No data found in '" + data_path + "'")
# # # if len(data_X) < batch_size: raise Exception("[!] Entire dataset size is less than the configured batch_size")

# # # data_X_ = np.stack([transform(np.array(Image.open(x).convert("RGB")),
# # #                                         256, 256,\
# # #                                         64, 64, False)\
# # #                                         for x in data_X])
# # # buffer_size=data_X_.shape[0]
# # # data_X_tf = tf.data.Dataset.from_tensor_slices(data_X_).shuffle(buffer_size).batch(batch_size)

# # # x=0

# # # import math

# # # def conv_out_size_same(size, stride):
# # #   return int(math.ceil(float(size) / float(stride)))

# # # s_h, s_w = 28, 28
# # # print(s_h, s_w)
# # # s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
# # # print(s_h2, s_w2)
# # # s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
# # # print(s_h4, s_w4)
# # # s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
# # # print(s_h8, s_w8)
# # # s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
# # # print(s_h16, s_w16)



# # # import math
# # # def conv_out_size_same(size, stride):
# # #   return int(math.ceil(float(size) / float(stride)))
# # # out=64
# # # for scaler in range(2, 10, 2):
# # #   out = conv_out_size_same(out, 2)
# # #   print(out)

# # # import tensorflow as tf
# # # from tensorflow.keras import layers # pyright: ignore
# # # import math

# # # def conv_out_size_same(size, stride):
# # #   return int(math.ceil(float(size) / float(stride)))

# # # def conv_cond_concat(x, y):
# # #   """Concatenate conditioning vector on feature map axis."""
# # #   x_shapes = x.get_shape()
# # #   y_shapes = y.get_shape()
# # #   return tf.concat([
# # #     x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

# # # output_height = output_width = 64
# # # gf_dim=64
# # # z_dim=100
# # # c_dim=3
# # # gfc_dim=1024
# # # batch_size=128
# # # y_dim=100
# # # y=tf.zeros([batch_size, y_dim], dtype=tf.float32)
# # # z = tf.zeros([1, z_dim], dtype=tf.float32)
# # # s_h, s_w = output_height, output_width
# # # s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
# # # s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
# # # s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
# # # s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)




# # # if not y_dim:
# # #   s_h, s_w = output_height, output_width
# # #   s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
# # #   s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
# # #   s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
# # #   s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

# # #   model = tf.keras.Sequential()
  
# # #   model.add(layers.Dense(gf_dim*8*s_h16*s_w16, use_bias=True, input_shape=[None, z.get_shape()[1]]))
# # #   model.add(layers.BatchNormalization())
# # #   model.add(layers.LeakyReLU())

# # #   model.add(layers.Reshape((s_h16, s_w16, gf_dim * 8)))
# # #   assert model.output_shape == (None, s_h16, s_w16, gf_dim * 8)  # Note: None is the batch size

# # #   model.add(layers.Conv2DTranspose(gf_dim*4, (5, 5), strides=(2, 2), padding='same', use_bias=True))
# # #   assert model.output_shape == (None, s_h8, s_w8, gf_dim*4)
# # #   model.add(layers.BatchNormalization())
# # #   model.add(layers.LeakyReLU())

# # #   model.add(layers.Conv2DTranspose(gf_dim*2, (5, 5), strides=(2, 2), padding='same', use_bias=True))
# # #   assert model.output_shape == (None, s_h4, s_w4, gf_dim*2)
# # #   model.add(layers.BatchNormalization())
# # #   model.add(layers.LeakyReLU())

# # #   model.add(layers.Conv2DTranspose(gf_dim, (5, 5), strides=(2, 2), padding='same', use_bias=True))
# # #   assert model.output_shape == (None, s_h2, s_w2, gf_dim)
# # #   model.add(layers.BatchNormalization())
# # #   model.add(layers.LeakyReLU())

# # #   model.add(layers.Conv2DTranspose(c_dim, (5, 5), strides=(2, 2), padding='same', use_bias=True, activation='tanh'))
# # #   assert model.output_shape == (None, s_h, s_w, c_dim)

# # #   # return model

# # # else:
# # #   s_h, s_w = output_height, output_width
# # #   s_h2, s_h4 = int(s_h/2), int(s_h/4)
# # #   s_w2, s_w4 = int(s_w/2), int(s_w/4)

# # #   yb = tf.reshape(y, [batch_size, 1 , 1, y_dim])
# # #   z = tf.concat([z, y], 1)

# # #   model = tf.keras.Sequential()

# # #   model.add(tf.keras.layers.Dense(gfc_dim, use_bias=True, input_shape=[None, z.get_shape()[1]]))
# # #   model.add(tf.keras.layers.BatchNormalization())
# # #   model.add(tf.keras.layers.LeakyReLU())
# # #   model.add(tf.keras.layers.Concatenate(axis=1)[model.output, y])
# # #   assert model.output_shape == (None, gfc_dim, y.get_shape()[1])

# # #   model.add(tf.keras.layers.Dense(gf_dim*2*s_h4*s_w4, use_bias=True, ))
# # #   model.add(tf.keras.layers.BatchNormalization())
# # #   model.add(tf.keras.layers.LeakyReLU())
# # #   model.add(tf.keras.layers.Reshape(s_h4, s_w4, gf_dim * 2))
# # #   model.add(conv_cond_concat(x=model.output, y=yb))
# # #   assert model.output_shape == (None, s_h4, s_w4, gf_dim * 2)

# # #   model.add(tf.keras.layers.Conv2DTranspose(gf_dim * 2, (5, 5), strides=(2, 2), padding='same', use_bias=True))
# # #   model.add(tf.keras.layers.BatchNormalization())
# # #   model.add(tf.keras.layers.LeakyReLU())
# # #   model.add(conv_cond_concat(model.output, yb))
# # #   assert model.output_shape == (None, s_h2, s_w2, gf_dim*2)

# # #   model.add(tf.keras.layers.Conv2DTranspose(c_dim, (5, 5), strides=(2, 2), padding='same', use_bias=True, activation="sigmoid"))
# # #   assert model.output==(None, s_h, s_w, c_dim)
# # #   # return model



# # # x=0

# # # import tensorflow as tf
# # # df_dim = 64
# # # batch_size=64
# # # image = tf.zeros([64, 64, 3], dtype=tf.float32)
# # # model = tf.keras.Sequential()
# # # model.add(tf.keras.layers.Conv2D(df_dim, (5,5), (2,2), "same", use_bias=True, input_shape=image.shape,
# # #                                   kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None),
# # #                                   bias_initializer=tf.keras.initializers.Constant(value=0),))
# # # model.add(tf.keras.layers.LeakyReLU())
# # # # model.add(tf.keras.layers.Dropout(0.3)) - repo not doing this, but paper do this

# # # model.add(tf.keras.layers.Conv2D(df_dim*2, (5,5), (2,2,), "same", use_bias=True,
# # #                                   kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None),
# # #                                   bias_initializer=tf.keras.initializers.Constant(value=0),))
# # # model.add(tf.keras.layers.LeakyReLU())

# # # model.add(tf.keras.layers.Conv2D(df_dim*4, (5,5), (2,2,), "same", use_bias=True,
# # #                                   kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None),
# # #                                   bias_initializer=tf.keras.initializers.Constant(value=0),))
# # # model.add(tf.keras.layers.LeakyReLU())

# # # model.add(tf.keras.layers.Conv2D(df_dim*8, (5,5), (2,2,), "same", use_bias=True,
# # #                                   kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None),
# # #                                   bias_initializer=tf.keras.initializers.Constant(value=0),))
# # # model.add(tf.keras.layers.LeakyReLU())

# # # model.add(tf.keras.layers.Flatten())
# # # model.add(tf.keras.layers.Dense(1, use_bias=True, 
# # #                                 kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None),
# # #                                 bias_initializer=tf.keras.initializers.Constant(value=0),))
# # # h4_logits = model.output
# # # model.add(tf.keras.layers.Activation(tf.nn.sigmoid))











# # # #region inspect argparse to json error - Namespace is not subscriptable
# # # import os
# # # import json
# # # import numpy as np
# # # import argparse


# # # parser = argparse.ArgumentParser(description="Train, Test or Infer DCGAN")

# # # parser.add_argument("--epoch", type=int, default=25, help="Epoch to train [25]")
# # # parser.add_argument("--learning-rate", type=float, default=0.0002, help="Learning rate of for adam [0.0002]")
# # # parser.add_argument("--beta1", type=float, default=0.5, help="Momentum term of adam [0.5]")
# # # parser.add_argument("--train-size", default=np.inf, help="The size of train images [np.inf]")
# # # parser.add_argument("--batch-size", type=int, default=64, help="The size of batch images [64]")
# # # parser.add_argument("--input-height", type=int, default=108, help="The size of image to use (will be center cropped). [108]")
# # # parser.add_argument("--input-width", type=int, default=None, help="The size of image to use (will be center cropped). If None, same value as input_height [None]")
# # # parser.add_argument("--output-height", type=int, default=64, help="The size of the output images to produce [64]")
# # # parser.add_argument("--output-width", type=int, default=None, help="The size of the output images to produce. If None, same value as output_height [None]")
# # # parser.add_argument("--dataset", type=str, default="celebA", help="The name of dataset [celebA, mnist, lsun]")
# # # parser.add_argument("--input-fname-pattern", type=str, default="*.jpg", help="Glob pattern of filename of input images [*]")
# # # parser.add_argument("--data-dir", type=str, default="./data", help="path to datasets [e.g. $HOME/data]")
# # # parser.add_argument("--out-dir", type=str, default="./out", help="Root directory for outputs [e.g. $HOME/out]")
# # # parser.add_argument("--out-name", type=str, default="", help="Folder (under out_root_dir) for all outputs. Generated automatically if left blank []")
# # # parser.add_argument("--checkpoint-dir", type=str, default="checkpoint", help="Folder (under out_root_dir/out_name) to save checkpoints [checkpoint]")
# # # parser.add_argument("--sample-dir", type=str, default="samples", help="Folder (under out_root_dir/out_name) to save samples [samples]")
# # # parser.add_argument("--train", type=bool, default=False, help="True for training, False for testing [False]")
# # # parser.add_argument("--crop", type=bool, default=False, help="True for training, False for testing [False]")
# # # parser.add_argument("--visualize", type=bool, default=False, help="True for visualizing, False for nothing [False]")
# # # parser.add_argument("--export", type=bool, default=False, help="True for exporting with new batch size")
# # # parser.add_argument("--freeze", type=bool, default=False, help="True for exporting with new batch size")
# # # parser.add_argument("--max-to-keep", type=int, default=1, help="maximum number of checkpoints to keep")
# # # parser.add_argument("--sample-freq", type=int, default=200, help="sample every this many iterations")
# # # parser.add_argument("--ckpt-freq", type=int, default=200, help="save checkpoint every this many iterations")
# # # parser.add_argument("--z-dim", type=int, default=100, help="dimensions of z")
# # # parser.add_argument("--z-dist", type=str, default="uniform_signed", help="'normal01' or 'uniform_unsigned' or uniform_signed")
# # # parser.add_argument("--G-img-sum", type=bool, default=False, help="Save generator image summaries in log")
# # # parser.add_argument("--generate-test-images", type=int, default=100, help="Number of images to generate during test. [100]")


# # # args = parser.parse_args()
# # # print(args)
  

# # # with open(os.path.join(args.out_dir, 'arguments.json'), 'w') as f:
# # #     arguments_dict = vars(args)
# # #     json.dump(arguments_dict, f, indent=4, sort_keys=True, ensure_ascii=False)
# # # #endregion
# # #
# # # # region expand path
# # # import os
# # # x = "data"
# # # print(os.path.expanduser(os.path.expandvars(x)))

# # # #endregion
