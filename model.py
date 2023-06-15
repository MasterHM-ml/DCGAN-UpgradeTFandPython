from __future__ import division
from __future__ import print_function

import os
import time
import json
import math
import numpy as np
import tensorflow as tf
from glob import glob
from PIL import Image

import logging
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm

import matplotlib.pyplot as plt

from utils import save_images, transform


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


def gen_random(mode, size):
    if mode == 'normal01': return np.random.normal(0, 1, size=size)
    if mode == 'uniform_signed': return np.random.uniform(-1, 1, size=size)
    if mode == 'uniform_unsigned': return np.random.uniform(0, 1, size=size)


class Losses:
    def __init__(self):
        self.running_loss = []
        self.epoch_loss = []


class ModelLosses:
    def __init__(self):
        self.discriminator = Losses()
        self.generator = Losses()


class DCGAN(object):
    def __init__(self, input_height=108, input_width=108, crop=True,
                 batch_size=64, output_height=64, output_width=64,
                 z_dim=100, gf_dim=64, df_dim=64, train=True, c_dim=3,
                 gfc_dim=1024, dfc_dim=1024, dataset_name='default',
                 max_to_keep=1, early_stop_count=20, checkpoint_prefix="checkpoint",
                 input_fname_pattern='*.jpg', checkpoint_dir='ckpts', sample_dir='samples', out_dir='./out',
                 data_dir='./data'):
        """
        Args:
        batch_size: The size of batch. Should be specified before training.
        z_dim: (optional) Dimension of dim for Z. [100]
        gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
        df_dim: (optional) Dimension of discriminator filters in first conv layer. [64]
        gfc_dim: (optional) Dimension of gen units for fully connected layer. [1024]
        dfc_dim: (optional) Dimension of discriminator units for fully connected layer. [1024]
        """
        self.z = None
        self.num_of_batches = None
        self.num_of_images_in_dataset = None
        self.buffer_size = None

        self.generator_model = None
        self.discriminator_model = None
        self.d_optim = None
        self.g_optim = None
        self.losses = None
        self.cross_entropy = None
        self.checkpoint = None

        self.crop = crop
        self.do_training = train

        self.batch_size = batch_size

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.c_dim = c_dim
        self.z_dim = z_dim
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.input_fname_pattern = input_fname_pattern
        self.out_dir = out_dir
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = checkpoint_prefix
        self.early_stop_count = early_stop_count

        self.checkpoint_best_model = os.path.join(self.checkpoint_dir, "best_model")
        self.checkpoint_best_gen = os.path.join(self.checkpoint_best_model, "gen")
        self.checkpoint_best_dis = os.path.join(self.checkpoint_best_model, "dis")
        self.max_to_keep = max_to_keep
        self.sample_dir = sample_dir
        self.image_io_json = os.path.join(self.checkpoint_dir, "image_io.json")

        if self.do_training:
            if self.dataset_name in ["mnist", "fashion_mnist", "cifar10", "cifar100"]:
                self.data_X, self.data_Y = self.load_builtin_dataset()
            else:
                self.data_X, self.data_Y = self.load_custom_dataset()

        self.build_model()

    def build_model(self):
        if not self.do_training:
            if len(glob((os.path.join(self.checkpoint_dir, "checkpoint*")))) == 0:
                raise ValueError("Can't find checkpoints in %s. Did you even trained model on %s today?"
                                 " Pass '--out-dir' from previous date e.g. ./out/cifar100_2023-06-13 if you want to generate images with older models" % (
                    self.checkpoint_dir, self.dataset_name,))
            if not os.path.exists(self.image_io_json):
                raise ValueError(
                    "Can't find crucial image_io data file under %s that was saved during training"
                    % self.image_io_json)

            with open(self.image_io_json, "r") as ff:
                ff = json.load(ff)
                self.input_height = self.input_width = ff["input_height_width"]
                self.output_height = self.output_width = ff["output_height_width"]
                self.c_dim = ff["channel"]
                image_dims = [self.input_height, self.input_width, self.c_dim]
        else:
            if self.crop:
                image_dims = [self.output_height, self.output_width, self.c_dim]
            else:
                image_dims = [self.input_height, self.input_width, self.c_dim]

        self.z = tf.Variable(tf.zeros([self.batch_size, self.z_dim]), dtype=tf.float32)
        self.generator_model = self.generator(self.z, image_dims)
        self.discriminator_model = self.discriminator(image_dims)
        self.cross_entropy = tf.keras.losses.get({
            "class_name": "BinaryCrossentropy",
            "config": {"from_logits": True}
        })
        self.losses: ModelLosses = ModelLosses()
        self.d_optim = tf.keras.optimizers.Adam()  # .minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.keras.optimizers.Adam()  # .minimize(self.g_loss, var_list=self.g_vars)
        self.checkpoint = tf.train.Checkpoint(g_optim=self.g_optim,
                                              d_optim=self.d_optim,
                                              generator_model=self.generator_model,
                                              discriminator_model=self.discriminator_model)

    def train(self, config):
        early_stop_count_tracker = 0
        self.d_optim = tf.keras.optimizers.Adam(
            config.learning_rate)
        self.g_optim = tf.keras.optimizers.Adam(
            config.learning_rate)
        sample_z = tf.random.normal([self.batch_size, self.z_dim])
        self.num_of_batches = self.num_of_images_in_dataset // self.batch_size

        start_time = time.time()
        # TODO resume training
        with open(self.image_io_json, "w") as ff:
            json.dump({
                "input_height_width": self.input_height,
                "output_height_width": self.output_height,
                "channel": self.c_dim
            }, ff)
        with logging_redirect_tqdm():
            minimum_loss = np.Inf
            batch_tracker = 0
            for epoch in trange(config.epoch):
                for idx, batch_images in enumerate(self.data_X):
                    batch_tracker+=1
                    gl, dl = self.train_step(batch_images)
                    self.losses.discriminator.running_loss.append(tf.reduce_mean(dl).numpy().item())
                    self.losses.generator.running_loss.append(tf.reduce_mean(gl).numpy().item())
                    if np.mod(idx, config.logging_frequency) == 0:
                        logging.info("[Epoch: %2d/%2d] [Batch: %4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f"
                                     % (epoch + 1, config.epoch, batch_tracker, self.num_of_batches * self.batch_size,
                                        time.time() - start_time,
                                        np.mean(self.losses.discriminator.running_loss).item(),
                                        np.mean(self.losses.generator.running_loss).item()))

                    # if idx == 2:
                    #     break

                self.losses.generator.epoch_loss.append(np.mean(self.losses.generator.running_loss).item())
                self.losses.discriminator.epoch_loss.append(np.mean(self.losses.discriminator.running_loss).item())
                self.losses.generator.running_loss.clear()
                self.losses.discriminator.running_loss.clear()
                logging.info("[Epoch: %2d/%2d] time: %4.4f, d_loss: %.8f, g_loss: %.8f"
                             % (epoch + 1, config.epoch,
                                time.time() - start_time, self.losses.discriminator.epoch_loss[-1],
                                self.losses.generator.epoch_loss[-1]))

                if np.mod(epoch, config.sample_freq) == 0:
                    self.generate_and_save_images(self.generator_model, epoch + 1, sample_z)
                if np.mod(epoch, config.ckpt_freq) == 0:
                    if len(glob(self.checkpoint_prefix + "*")) > (self.max_to_keep * 2):
                        os.remove(self.checkpoint_prefix)
                        home_dir = os.getcwd()
                        os.chdir(self.checkpoint_dir)
                        list_old_ckpt = glob(self.checkpoint_prefix.split("/")[-1] + "*")
                        list_old_ckpt.sort(key= lambda x: int(x[11:].split(".")[0]))
                        os.remove(list_old_ckpt[0])
                        os.remove(list_old_ckpt[1])
                        os.chdir(home_dir)
                    self.checkpoint.save(file_prefix=self.checkpoint_prefix)

                if (epoch - early_stop_count_tracker) > self.early_stop_count:
                        logging.warning("QUIT TRAINING - Early stopping count reached. %d", self.early_stop_count)
                        break
                if self.losses.generator.epoch_loss[-1] < minimum_loss:
                    early_stop_count_tracker = epoch
                    minimum_loss = self.losses.generator.epoch_loss[-1]
                    _ = [os.remove(old_best) for old_best in glob(f"{self.checkpoint_best_model}*")]
                    self.checkpoint.save(file_prefix=self.checkpoint_best_model)
                    # tf.saved_model.save(self.generator_model, self.checkpoint_best_gen)
                    # tf.saved_model.save(self.discriminator_model, self.checkpoint_best_dis)
                
        logging.info("Training Completed!!!!")

    def discriminator(self, image_dims, ):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(self.df_dim, (5, 5), (2, 2), "same", use_bias=True, input_shape=image_dims,
                                         kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02,
                                                                                                  seed=None),
                                         bias_initializer=tf.keras.initializers.Constant(value=0), ))
        model.add(tf.keras.layers.LeakyReLU())
        # model.add(tf.keras.layers.Dropout(0.3)) - repo not doing this, but paper do this

        model.add(tf.keras.layers.Conv2D(self.df_dim * 2, (5, 5), (2, 2,), "same", use_bias=True,
                                         kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02,
                                                                                                  seed=None),
                                         bias_initializer=tf.keras.initializers.Constant(value=0), ))
        model.add(tf.keras.layers.LeakyReLU())

        model.add(tf.keras.layers.Conv2D(self.df_dim * 4, (5, 5), (2, 2,), "same", use_bias=True,
                                         kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02,
                                                                                                  seed=None),
                                         bias_initializer=tf.keras.initializers.Constant(value=0), ))
        model.add(tf.keras.layers.LeakyReLU())

        model.add(tf.keras.layers.Conv2D(self.df_dim * 8, (5, 5), (2, 2,), "same", use_bias=True,
                                         kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02,
                                                                                                  seed=None),
                                         bias_initializer=tf.keras.initializers.Constant(value=0), ))
        model.add(tf.keras.layers.LeakyReLU())

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1, use_bias=True, name="h4",
                                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02,
                                                                                              seed=None),
                                        bias_initializer=tf.keras.initializers.Constant(value=0), ))
        # h4_logits = model.get_layer('h4')
        # TODO whether to comment/uncomment - verify from old train method
        # either old repo uses h4 or sigmoid
        # model.add(tf.keras.layers.Activation(tf.nn.sigmoid))
        # return model, h4_logits
        return model

    def generator(self, z, image_dims):
        s_h, s_w = image_dims[0], image_dims[1]
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Dense(self.gf_dim * 8 * s_h16 * s_w16, use_bias=True, input_shape=[z.get_shape()[1]],
                                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02,
                                                                                              seed=None),
                                        bias_initializer=tf.keras.initializers.Constant(value=0), ))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.ReLU())

        model.add(tf.keras.layers.Reshape((s_h16, s_w16, self.gf_dim * 8)))
        assert model.output_shape == (None, s_h16, s_w16,
                                      self.gf_dim * 8), "Shape mismatch, use input shape which is 2^x e.g. 32, 64, " \
                                                        "128 etc."  # Note: None is the batch size

        model.add(
            tf.keras.layers.Conv2DTranspose(self.gf_dim * 4, (5, 5), strides=(2, 2), padding='same', use_bias=True,
                                            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02,
                                                                                                  seed=None),
                                            bias_initializer=tf.keras.initializers.Constant(value=0), ))
        assert model.output_shape == (
            None, s_h8, s_w8, self.gf_dim * 4), "Shape mismatch, use input shape which is 2^x e.g. 32, 64, 128 etc."
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.ReLU())

        model.add(
            tf.keras.layers.Conv2DTranspose(self.gf_dim * 2, (5, 5), strides=(2, 2), padding='same', use_bias=True,
                                            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02,
                                                                                                  seed=None),
                                            bias_initializer=tf.keras.initializers.Constant(value=0), ))
        assert model.output_shape == (
            None, s_h4, s_w4, self.gf_dim * 2), "Shape mismatch, use input shape which is 2^x e.g. 32, 64, 128 etc."
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.ReLU())

        model.add(tf.keras.layers.Conv2DTranspose(self.gf_dim, (5, 5), strides=(2, 2), padding='same', use_bias=True,
                                                  kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0,
                                                                                                        stddev=0.02,
                                                                                                        seed=None),
                                                  bias_initializer=tf.keras.initializers.Constant(value=0), ))
        assert model.output_shape == (
            None, s_h2, s_w2, self.gf_dim), "Shape mismatch, use input shape which is 2^x e.g. 32, 64, 128 etc."
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        model.add(
            tf.keras.layers.Conv2DTranspose(self.c_dim, (5, 5), strides=(2, 2), padding='same', use_bias=True,
                                            activation='tanh',
                                            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02,
                                                                                                  seed=None),
                                            bias_initializer=tf.keras.initializers.Constant(value=0), ))
        assert model.output_shape == (
            None, s_h, s_w, self.c_dim), "Shape mismatch, use input shape which is 2^x e.g. 32, 64, 128 etc."

        return model

    def load_builtin_dataset(self):
        logging.info("Loading built-in dataset, input_height and input_width will be reset")
        (train_images, _), (_, _) = getattr(tf.keras.datasets, self.dataset_name).load_data()
        if train_images.shape[1] < 32: train_images = tf.pad(train_images, [[0, 0], [2, 2], [2, 2]]).numpy()
        self.input_height = self.input_width = train_images[0].shape[0]
        if train_images[0].shape[-1] == 3:  # either shape will be HxW or HxWx3
            self.c_dim = 3
        else:
            self.c_dim = 1
        self.num_of_images_in_dataset = train_images.shape[0]
        train_images = train_images.reshape(train_images.shape[0], self.input_width, self.input_height,
                                            self.c_dim).astype('float32')
        # train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
        train_images = train_images / 255  # Normalize the images to [-1, 1]
        self.buffer_size = train_images.shape[0]
        test_dataset = np.ones(train_images.shape[0])
        train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(self.buffer_size).batch(
            self.batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices(test_dataset).shuffle(self.buffer_size).batch(self.batch_size)
        return train_dataset, test_dataset

    def load_custom_dataset(self):
        data_path = os.path.join(self.data_dir, self.dataset_name, self.input_fname_pattern)
        logging.info("loading custom data from %s." % data_path)
        if self.c_dim is None:
            raise ValueError(
                "with custom data, c-dim argument is required. 1 if your data is grayscale or 3 if your dataset is "
                "RGB images")
        logging.warning(
            "Please make sure all images are either RGB (3 channels) or grayscale (1 channels). Got argument 'c_dim'=%d"
            % self.c_dim)
        path_to_images = glob(data_path)
        if len(path_to_images) == 0: raise Exception("[!] No data found in '" + data_path + "'")
        if len(path_to_images) < self.batch_size: raise Exception(
            "[!] Entire dataset size is less than the configured batch_size")

        if Image.open(path_to_images[0]).size != (self.input_width, self.input_height):
            logging.warning("[!] Image dim, and provided input_height, input_width are not same.")
        if self.c_dim == 1:
            data_x_np = np.stack([transform(np.array(Image.open(x).convert('L')),
                                            self.input_height, self.input_width,
                                            self.output_height, self.output_width, self.crop)
                                  for x in path_to_images])
        elif self.c_dim == 3:
            data_x_np = np.stack([transform(np.array(Image.open(x).convert('RGB')),
                                            self.input_height, self.input_width,
                                            self.output_height, self.output_width, self.crop)
                                  for x in path_to_images])
        else:
            raise Exception("[!] Unknown color dimension. Got argument 'c_dim'=%d" % self.c_dim)
        self.num_of_images_in_dataset = len(path_to_images)
        self.buffer_size = data_x_np.shape[0]
        train_dataset_y = tf.data.Dataset.from_tensor_slices(np.ones(data_x_np.shape[0])).shuffle(
            self.buffer_size).batch(self.batch_size)
        train_dataset_x = tf.data.Dataset.from_tensor_slices(data_x_np).shuffle(self.buffer_size).batch(self.batch_size)
        return train_dataset_x, train_dataset_y

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([self.batch_size, self.z_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as gen_tape_2, tf.GradientTape() as disc_tape:
            generated_images = self.generator_model(noise, training=True)

            real_output = self.discriminator_model(images, training=True)
            fake_output = self.discriminator_model(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        # updating generator twice a step - as done in original repo
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator_model.trainable_variables)
        self.g_optim.apply_gradients(zip(gradients_of_generator, self.generator_model.trainable_variables))
        gradients_of_generator = gen_tape_2.gradient(gen_loss, self.generator_model.trainable_variables)
        self.g_optim.apply_gradients(zip(gradients_of_generator, self.generator_model.trainable_variables))

        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator_model.trainable_variables)
        self.d_optim.apply_gradients(zip(gradients_of_discriminator, self.discriminator_model.trainable_variables))

        return gen_loss, disc_loss

    def generate_and_save_images(self, model, epoch, test_input, draw_loss_graph=True):
        predictions = model(test_input, training=False)
        predictions = np.array(predictions.numpy()*255, dtype=np.uint8)
        if (not draw_loss_graph):
            if predictions.shape[-1]==3:
                [Image.fromarray(predictions[i]).save(os.path.join(self.sample_dir, f"generated_{self.dataset_name}_{i}.jpg")) for i in range(predictions.shape[0])]
            else:
                [Image.fromarray(np.squeeze(predictions[i]), "L").save(os.path.join(self.sample_dir, f"generated_{self.dataset_name}_{i}.jpg")) for i in range(predictions.shape[0])]
            save_images(predictions, (self.output_height, self.output_width),
                        os.path.join(self.sample_dir, "big_tiff_image.tiff"))
        _ = plt.figure(figsize=(4, 4))
        if self.c_dim == 1:
            cmap_ = "gray"
        else:
            cmap_ = None
        for i in range(predictions.shape[0]):
            if i == 16:  # TODO fix it - subplots are 16 only, will output 16 images only
                break
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i], cmap=cmap_)
            plt.axis('off')
        plt.savefig(os.path.join(self.sample_dir, f'image_at_{epoch}.png'))
        plt.close()
        # plt.show()

        if draw_loss_graph:
            _ = plt.figure(figsize=(12, 8))
            plt.plot(self.losses.discriminator.epoch_loss, label="Discriminator Loss", linewidth=2, marker="*")
            plt.plot(self.losses.generator.epoch_loss, label="Generator Loss", linewidth=2, marker="*")
            plt.legend()
            plt.savefig(os.path.join(self.sample_dir, "losses.png"))
            plt.close()

    def load_and_generate_images(self, args):
        logging.info(" [*] Reading checkpoints... %s" % args.checkpoint_dir)
        try:
            if args.load_best_model_only:
                self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_best_model))
                logging.info("Loaded best model from %s" % args.checkpoint_dir)
            else:
                self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
                logging.info("Loaded latest model from %s" % args.checkpoint_dir)
        except Exception as e:
            logging.info(f" [*] Failed to find checkpoint at {self.checkpoint_best_gen}")
            raise Exception(e)
        z = tf.random.normal([args.generate_test_images, self.z_dim])
        self.generate_and_save_images(self.generator_model, f"generated_{self.dataset_name}_", z, draw_loss_graph=False)
        return True
