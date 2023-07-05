import os
import time
import cv2
import json
import math
from math import floor
import numpy as np
import tensorflow as tf
from glob import glob

import cv2
from PIL import Image

import logging
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm

import matplotlib.pyplot as plt

from utils import transform


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
                 z_dim=100, gf_dim=64, df_dim=64, train=True, retrain=False, c_dim=3, load_model_dir=None,
                 gfc_dim=1024, dfc_dim=1024, dataset_name='default',
                 checkpoint_prefix="checkpoint",
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
        self.path_to_images = None
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
        self.checkpointer = None

        self.crop = crop
        self.do_training = train
        self.do_retraining = retrain

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
        self.sample_dir = sample_dir
        self.load_model_dir = load_model_dir
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = checkpoint_prefix

        self.checkpoint_best_model = os.path.join(self.checkpoint_dir, "best_model", "best_model")
        self.out_media_epoch_path = os.path.join(self.sample_dir, "media", "epoch")
        self.out_media_collage_path = os.path.join(self.sample_dir, "media", "collage")
        self.out_sample_loss_path = os.path.join(self.sample_dir, "losses")
        if not os.path.exists(self.out_media_collage_path): os.makedirs(self.out_media_collage_path)
        if not os.path.exists(self.out_media_epoch_path): os.makedirs(self.out_media_epoch_path)
        if not os.path.exists(self.out_sample_loss_path): os.makedirs(self.out_sample_loss_path)
        if self.do_retraining:
            self.image_io_json = os.path.join(self.load_model_dir, "image_io.json")
        else:
            self.image_io_json = os.path.join(self.checkpoint_dir, "image_io.json")

        if self.do_training or self.do_retraining:
            self.load_metadata()
            self.data_yielder = self.load_custom_dataset()

        self.build_model()

    def build_model(self):
        if self.do_training:
            if self.crop:
                image_dims = [self.output_height, self.output_width, self.c_dim]
            else:
                image_dims = [self.input_height, self.input_width, self.c_dim]
        else:
            if not os.path.exists(self.image_io_json):
                raise ValueError(
                    "Can't find crucial image_io data file under %s that was saved during training/previous_training"
                    % self.image_io_json)

            with open(self.image_io_json, "r") as ff:
                ff = json.load(ff)
                self.input_height = self.input_width = ff["input_height_width"]
                self.output_height = self.output_width = ff["output_height_width"]
                self.c_dim = ff["channel"]
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
        self.checkpointer = tf.train.Checkpoint(g_optim=self.g_optim,
                                                d_optim=self.d_optim,
                                                generator_model=self.generator_model,
                                                discriminator_model=self.discriminator_model)

    def train(self, config):
        sample_z = tf.random.uniform(shape=[self.batch_size, self.z_dim], minval=0, maxval=1)
        self.num_of_batches = self.num_of_images_in_dataset // self.batch_size

        g_lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            config.g_learning_rate,
            decay_steps=self.num_of_batches,
            decay_rate=0.96,
            staircase=True)
        d_lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            config.d_learning_rate,
            decay_steps=self.num_of_batches,
            decay_rate=0.8,
            staircase=True)
        self.d_optim = tf.keras.optimizers.Adam(
            learning_rate=d_lr_scheduler
        )
        self.g_optim = tf.keras.optimizers.Adam(
            learning_rate=g_lr_scheduler)

        start_time = time.time()
        with open(os.path.join(self.checkpoint_dir, "image_io.json"), "w") as ff:
            json.dump({
                "input_height_width": self.input_height,
                "output_height_width": self.output_height,
                "channel": self.c_dim
            }, ff)
        
        start_time = time.time()
        # TODO resume training
        with logging_redirect_tqdm():
            minimum_loss = np.Inf
            for epoch in trange(config.epoch):
                self.data_yielder = self.load_custom_dataset()  # * reset generator index after each epoch
                for idx, batch_images in enumerate(self.data_yielder):
                    batch_tracker += 1
                    gl, dl = self.train_step(batch_images.as_numpy_iterator().next())
                    '''
                    thanks BingAI - solved TypeError: Inputs to a layer should be tensors got tensorflow.python.data.ops.dataset_ops._VariantDataset
                    '''
                    self.losses.discriminator.running_loss.append(tf.reduce_mean(dl).numpy().item())
                    self.losses.generator.running_loss.append(tf.reduce_mean(gl).numpy().item())
                    if np.mod(idx, config.logging_frequency) == 0:
                        logging.info("[Epoch: %2d/%2d] [Batch: %4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f"
                                     % (epoch + 1, config.epoch, idx, self.num_of_batches,
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
                # if np.mod(epoch, config.ckpt_freq) == 0:
                _ = [os.remove(old_model) for old_model in glob(f"{self.checkpoint_prefix}*")]
                self.checkpointer.save(file_prefix=self.checkpoint_prefix)

                if self.losses.generator.epoch_loss[-1] < minimum_loss:
                    minimum_loss = self.losses.generator.epoch_loss[-1]
                    _ = [os.remove(old_best) for old_best in glob(f"{self.checkpoint_best_model}*")]
                    self.checkpointer.save(file_prefix=self.checkpoint_best_model)

        logging.info("Training Completed!!!!")

    def retrain(self, config):
        logging.info(" [*] Reading checkpoints for retraining the model... %s" % config.load_model_dir)
        try:
            self.checkpointer.restore(tf.train.latest_checkpoint(config.load_model_dir))
            logging.info("Loaded latest model from %s" % config.load_model_dir)
        except Exception as e:
            logging.info(f" [*] Failed to find checkpoint at {config.load_model_dir}")
            raise Exception(e)
        self.train(config=config)

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

    def load_metadata(self, ):
        data_path = os.path.join(self.data_dir, self.dataset_name, self.input_fname_pattern)
        logging.info("loading custom data from %s." % data_path)
        logging.warning(
            "Please make sure all images are either RGB (3 channels). Got argument 'c-dim'=%d"
            % self.c_dim)
        self.path_to_images = glob(data_path)

        if len(self.path_to_images) == 0: raise Exception("[!] No data found in '" + data_path + "'")
        if len(self.path_to_images) < self.batch_size: raise Exception(
            "[!] Entire dataset size is less than the configured batch_size")

        if Image.open(self.path_to_images[0]).size != (self.input_width, self.input_height):
            logging.warning("[!] Image dim, and provided input_height, input_width are not same.")

        self.num_of_images_in_dataset = len(self.path_to_images)
        self.buffer_size = len(self.path_to_images)

    def load_custom_dataset(self):
        for batch_index in range(floor(int(self.num_of_images_in_dataset / self.batch_size))):
            image_list_for_batch = self.path_to_images[
                                   batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
            if self.c_dim == 3:
                data_x_np = np.stack([transform(np.array(Image.open(x).convert('RGB')),
                                                self.input_height, self.input_width,
                                                self.output_height, self.output_width, self.crop)
                                      for x in image_list_for_batch])
            else:
                raise Exception("[!] Unknown color dimension. Got argument 'c_dim'=%d" % self.c_dim)
            train_dataset_x = tf.data.Dataset.from_tensor_slices(data_x_np).shuffle(self.buffer_size).batch(
                self.batch_size)
            yield train_dataset_x

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

    def generate_and_save_images(self, model, img_save_name_indicator, test_input, draw_loss_graph=True):
        predictions = model(test_input, training=False)
        predictions = np.array(predictions.numpy() * 255,
                               dtype=np.uint8)  # TODO :: check un-normalization - do I need to dis-zero-center images as well
        save_string = None
        save_limit = None
        if type(img_save_name_indicator) == int:
            save_string = "epoch-%d" % img_save_name_indicator
            save_limit = 3
        elif type(img_save_name_indicator) == str:
            save_string = "%s-%s" % (img_save_name_indicator, self.dataset_name)
            save_limit = test_input.shape[0]
        for gen_image_indexer, gen_image in enumerate(predictions[:save_limit]):
            gen_image = cv2.cvtColor(gen_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(self.out_media_epoch_path, "%s-%d.jpg" % (save_string, gen_image_indexer)), gen_image)
        _ = plt.figure(figsize=(4, 4))
        if self.c_dim == 1:
            cmap_ = "gray"
        else:
            cmap_ = None
        for i in range(predictions.shape[0]):
            if i == 16:  # TODO fix it - subplots are 16 only, will output 16 images only
                break
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap=cmap_)
            # plt.imshow(predictions[i], cmap=cmap_)
            plt.axis('off')
        plt.savefig(os.path.join(self.out_media_collage_path, f'collage-{save_string}.png'))
        plt.close()

        if draw_loss_graph:
            pd.DataFrame(list(zip(self.losses.discriminator.epoch_loss, self.losses.generator.epoch_loss)),
                         columns=["DiscriminatorLoss", "GeneratorLoss"]).to_csv(
                os.path.join(self.out_sample_loss_path, "losses.csv"), index=False)
            _ = plt.figure(figsize=(12, 8))
            plt.plot(self.losses.discriminator.epoch_loss, label="Discriminator Loss", linewidth=2, marker="*")
            plt.plot(self.losses.generator.epoch_loss, label="Generator Loss", linewidth=2, marker="*")
            plt.legend()
            plt.savefig(os.path.join(self.out_sample_loss_path, "losses.png"))
            plt.close()

    def load_and_generate_images(self, args):
        logging.info(" [*] Reading checkpoints... %s" % args.checkpoint_dir)
        try:
            if args.load_best_model_only:
                self.checkpointer.restore(tf.train.latest_checkpoint(self.checkpoint_best_model))
                logging.info("Loaded best model from %s" % args.checkpoint_dir)
            else:
                self.checkpointer.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
                logging.info("Loaded latest model from %s" % args.checkpoint_dir)
        except Exception as e:
            logging.info(f" [*] Failed to find checkpoint at {args.checkpoint_dir}")
            raise Exception(e)
        z = tf.random.uniform([args.generate_test_images, self.z_dim], maxval=1)
        self.generate_and_save_images(self.generator_model, "generated", z, draw_loss_graph=False)
        return True
