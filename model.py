from __future__ import division
from __future__ import print_function
from typing import List, Dict
import os
import time
import math
import logging
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm

from glob import glob
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers  # pyright: ignore
import numpy as np
from six.moves import xrange # pyright: ignore

from ops import *
from utils import *

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

def gen_random(mode, size):
    if mode=='normal01': return np.random.normal(0,1,size=size)
    if mode=='uniform_signed': return np.random.uniform(-1,1,size=size)
    if mode=='uniform_unsigned': return np.random.uniform(0,1,size=size)




class Losses:
  def __init__(self):
    self.running_loss = List
    self.epoch_loss = List

class ModelLosses:
  def __init__(self):
    self.discriminator = Losses()
    self.generator = Losses()



class DCGAN(object):
  def __init__(self, tf_session, input_height=108, input_width=108, crop=True,
         batch_size=64, sample_num = 64, output_height=64, output_width=64,
         y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
         gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
         max_to_keep=1,
         input_fname_pattern='*.jpg', checkpoint_dir='ckpts', sample_dir='samples', out_dir='./out', data_dir='./data'):
    """

    Args:
      tf_session: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discriminator filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discriminator units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
    # self.tf_session = tf_session
    self.crop = crop

    self.batch_size = batch_size
    # self.sample_num = sample_num

    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width

    self.y_dim = y_dim
    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    # # batch normalization : deals with poor initialization helps gradient flow
    # self.d_bn1 = batch_norm(name='d_bn1')
    # self.d_bn2 = batch_norm(name='d_bn2')

    # if not self.y_dim:
    #   self.d_bn3 = batch_norm(name='d_bn3')

    # self.g_bn0 = batch_norm(name='g_bn0')
    # self.g_bn1 = batch_norm(name='g_bn1')
    # self.g_bn2 = batch_norm(name='g_bn2')

    # if not self.y_dim:
    #   self.g_bn3 = batch_norm(name='g_bn3')

    self.dataset_name = dataset_name
    self.data_dir = data_dir
    self.input_fname_pattern = input_fname_pattern
    self.out_dir = os.path.join(out_dir, self.dataset_name)

    self.checkpoint_dir = os.path.join(self.out_dir, checkpoint_dir)
    self.checkpoint_best_model = os.path.join(self.checkpoint_dir, "best_model")
    self.checkpoint_best_gen = os.path.join(self.checkpoint_best_model, "gen")
    self.checkpoint_best_dis = os.path.join(self.checkpoint_best_model, "dis")
    self.max_to_keep = max_to_keep
    self.sample_dir = os.path.join(self.out_dir, sample_dir)

    if self.dataset_name in ["mnist", "fashion_mnist", "cifar10", "cifar100"]:
      self.data_X, self.data_Y = self.load_builtin_dataset()
    else:
      self.data_X, self.data_Y = self.load_custom_dataset()
      
    # self.grayscale = (self.c_dim == 1)

    self.build_model()

  def build_model(self):
    # if self.y_dim:
    #   # self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
    #   self.y = tf.Variable(tf.zeros([self.batch_size, self.y_dim]), dtype=tf.float32)
    # else:
    #   self.y = None

    if self.crop:
      image_dims = [self.output_height, self.output_width, self.c_dim]
    else:
      image_dims = [self.input_height, self.input_width, self.c_dim]

    # self.inputs = tf.placeholder(
    #   tf.float32, [self.batch_size] + image_dims, name='real_images')
    # self.inputs = tf.Variable(tf.zeros([self.batch_size]+image_dims), dtype=tf.floats32) # SYNTAX ERROR
    # self.inputs = tf.Variable(tf.zeros([self.batch_size]+image_dims), dtype=tf.float32)

    # inputs = self.inputs

    # self.z = tf.placeholder(
    #   tf.float32, [None, self.z_dim], name='z')
    self.z = tf.Variable(tf.zeros([None, self.z_dim]), dtype=tf.float32)
    self.z_sum = histogram_summary("z", self.z)

    self.generator_model = self.generator(self.z)
    self.discriminator_model = self.discriminator(image_dims)
    # self.discriminator_model, self.D_logits = self.discriminator(image_dims=image_dims,)
    # self.discriminator_model_, self.D_logits_ = tf.keras.models.clone_model(self.discriminator_model), tf.identity(self.D_logits)
    
    # self.discriminator_model_summary = histogram_summary("d", self.discriminator_model)
    # self.d_loss_real = tf.reduce_mean(
    #   sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.discriminator_model)))
    # self.d_loss_fake = tf.reduce_mean(
    #   sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
    # self.g_loss = tf.reduce_mean(
    #   sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
    self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    self.losses: ModelLosses = ModelLosses()
    # self.d_loss_real = self.cross_entropy(self.D_logits, tf.ones_like(self.D_))

    self.d_optim = tf.keras.optimizers.Adam()# .minimize(self.d_loss, var_list=self.d_vars)
    self.g_optim = tf.train.optimizers.Adam()# .minimize(self.g_loss, var_list=self.g_vars)
    
    
    # self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    # self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
                          
    # self.d_loss = self.d_loss_real + self.d_loss_fake

    # self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    # self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    # t_vars = tf.trainable_variables()
    # self.d_vars = self.discriminator_model.trainable_variables()
    # self.g_vars = self.generator_model.trainable_variables()

    # self.d_vars = [var for var in t_vars if 'd_' in var.name]
    # self.g_vars = [var for var in t_vars if 'g_' in var.name]

    # self.saver = tf.train.Saver(max_to_keep=self.max_to_keep)
    self.checkpoint = tf.train.Checkpoint(g_optim=self.g_optim,
                                          d_optim=self.d_optim,
                                          generator_model=self.generator_model,
                                          discriminator_model=self.discriminator_model)

  def train(self, config):
    early_stop_count = 0
    self.d_optim = tf.keras.optimizers.Adam(config.learning_rate, beta1=config.beta1)# .minimize(self.d_loss, var_list=self.d_vars)
    self.g_optim = tf.train.optimizers.Adam(config.learning_rate, beta1=config.beta1)# .minimize(self.g_loss, var_list=self.g_vars)
    # try:
    #   tf.global_variables_initializer().run()
    # except:
    #   tf.initialize_all_variables().run()

    # if config.G_img_sum:
    #   self.g_sum = merge_summary([self.z_sum, self.d__sum, self.generator_model_summary, self.d_loss_fake_sum, self.g_loss_sum])
    # else:
    #   self.g_sum = merge_summary([self.z_sum, self.d__sum, self.d_loss_fake_sum, self.g_loss_sum])
    # self.discriminator_model_summary = merge_summary(
    #     [self.z_sum, self.discriminator_model_summary, self.d_loss_real_sum, self.d_loss_sum])
    # self.writer = SummaryWriter(os.path.join(self.out_dir, "logs"), self.tf_session.graph)

    sample_z = tf.random.normal([self.batch_size, self.z_dim])
    # sample_z = gen_random(config.z_dist, size=(self.batch_size , self.z_dim))
    # sample_inputs = self.data_X[0:self.batch_size]
    # sample_inputs = next(iter(self.data_X))
    # sample_labels = self.data_Y[0:self.batch_size]
    self.num_of_batches = self.data_X[0] // self.batch_size
    # if config.dataset == 'mnist':
    #   sample_inputs = self.data_X[0:self.batch_size]
    #   sample_labels = self.data_y[0:self.sample_num]
    # else:
    #   sample_files = self.data_X[0:self.sample_num]
    #   sample = [
    #       get_image(sample_file,
    #                 input_height=self.input_height,
    #                 input_width=self.input_width,
    #                 resize_height=self.output_height,
    #                 resize_width=self.output_width,
    #                 crop=self.crop,
    #                 grayscale=self.grayscale) for sample_file in sample_files]
    #   if (self.grayscale):
    #     sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
    #   else:
    #     sample_inputs = np.array(sample).astype(np.float32)
  
    # counter = 1
    start_time = time.time()
    # TODO resume training
    # could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    # if could_load:
    #   counter = checkpoint_counter
    #   print(" [*] Load SUCCESS")
    # else:
    #   print(" [!] Load failed...")
    with logging_redirect_tqdm():
      minimum_loss = np.Inf
      for epoch in trange(config.epoch):
        # if config.dataset == 'mnist':
        #   batch_idxs = min(len(self.data_X), config.train_size) // config.batch_size
        # else:      
        #   self.data_X = glob(os.path.join(
        #     config.data_dir, config.dataset, self.input_fname_pattern))
        #   np.random.shuffle(self.data_X)
        #   batch_idxs = min(len(self.data_X), config.train_size) // config.batch_size

        for idx, batch_images in enumerate(self.data_X):
          self.train_step(batch_images)

          if np.mod(idx, config.sample_freq) == 0:
            logging.info("[Epoch:[%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
              % (epoch, config.epoch, idx, self.num_of_batches,
                time.time() - start_time, self.losses.discriminator.epoch_loss[-1],\
                   self.losses.generator.epoch_loss[-1]))
            


        self.losses.generator.epoch_loss.append(np.mean(self.losses.generator.running_loss))
        self.losses.discriminator.epoch_loss.append(np.mean(self.losses.discriminator.running_loss))
        self.losses.generator.running_loss.clear()
        self.losses.discriminator.running_loss.clear()
        if np.mod(epoch, config.sample_freq) == 0:
          self.generate_and_save_images(self.generator_model, epoch+1, sample_z)

          # if config.dataset == 'mnist':
          #   batch_images = self.data_X[idx*config.batch_size:(idx+1)*config.batch_size]
          #   batch_labels = self.data_Y[idx*config.batch_size:(idx+1)*config.batch_size]
          # else:
          #   batch_files = self.data_X[idx*config.batch_size:(idx+1)*config.batch_size]
          #   batch = [
          #       get_image(batch_file,
          #                 input_height=self.input_height,
          #                 input_width=self.input_width,
          #                 resize_height=self.output_height,
          #                 resize_width=self.output_width,
          #                 crop=self.crop,
          #                 grayscale=self.grayscale) for batch_file in batch_files]
          #   if self.grayscale:
          #     batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
          #   else:
          #     batch_images = np.array(batch).astype(np.float32)

          # batch_z = gen_random(config.z_dist, size=[config.batch_size, self.z_dim]).astype(np.float32)

          # if config.dataset == 'mnist':
            # # Update D network
            # _, summary_str = self.tf_session.run([d_optim, self.discriminator_model_summary],
            #   feed_dict={ 
            #     self.inputs: batch_images,
            #     self.z: batch_z,
            #     self.y:batch_labels,
            #   })
            # self.writer.add_summary(summary_str, counter)

            # # Update G network
            # _, summary_str = self.tf_session.run([g_optim, self.g_sum],
            #   feed_dict={
            #     self.z: batch_z, 
            #     self.y:batch_labels,
            #   })
            # self.writer.add_summary(summary_str, counter)

            # # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
            # _, summary_str = self.tf_session.run([g_optim, self.g_sum],
            #   feed_dict={ self.z: batch_z, self.y:batch_labels })
            # self.writer.add_summary(summary_str, counter)
            
            # errD_fake = self.d_loss_fake.eval({
            #     self.z: batch_z, 
            #     self.y:batch_labels
            # })
            # errD_real = self.d_loss_real.eval({
            #     self.inputs: batch_images,
            #     self.y:batch_labels
            # })
            # errG = self.g_loss.eval({
            #     self.z: batch_z,
            #     self.y: batch_labels
            # })
          # else:
          #   # Update D network
          #   _, summary_str = self.tf_session.run([d_optim, self.discriminator_model_summary],
          #     feed_dict={ self.inputs: batch_images, self.z: batch_z })
          #   self.writer.add_summary(summary_str, counter)

          #   # Update G network
          #   _, summary_str = self.tf_session.run([g_optim, self.g_sum],
          #     feed_dict={ self.z: batch_z })
          #   self.writer.add_summary(summary_str, counter)

          #   # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
          #   _, summary_str = self.tf_session.run([g_optim, self.g_sum],
          #     feed_dict={ self.z: batch_z })
          #   self.writer.add_summary(summary_str, counter)
            
          #   errD_fake = self.d_loss_fake.eval({ self.z: batch_z })
          #   errD_real = self.d_loss_real.eval({ self.inputs: batch_images })
          #   errG = self.g_loss.eval({self.z: batch_z})

          # if np.mod(counter, config.sample_freq) == 0:
          #   if config.dataset == 'mnist':
          #     samples, d_loss, g_loss = self.tf_session.run(
          #       [self.sampler, self.d_loss, self.g_loss],
          #       feed_dict={
          #           self.z: sample_z,
          #           self.inputs: sample_inputs,
          #           self.y:sample_labels,
          #       }
          #     )
          #     save_images(samples, image_manifold_size(samples.shape[0]),
          #           './{}/train_{:08d}.png'.format(config.sample_dir, counter))
          #     print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
          #   else:
          #     try:
          #       samples, d_loss, g_loss = self.tf_session.run(
          #         [self.sampler, self.d_loss, self.g_loss],
          #         feed_dict={
          #             self.z: sample_z,
          #             self.inputs: sample_inputs,
          #         },
          #       )
          #       save_images(samples, image_manifold_size(samples.shape[0]),
          #             './{}/train_{:08d}.png'.format(config.sample_dir, counter))
          #       print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
          #     except:
          #       print("one pic error!...")

        if np.mod(epoch, config.ckpt_freq) == 0:
          if os.listdir(self.checkpoint_dir) > self.max_to_keep:
            old_ckpt = sorted(os.listdir(self.checkpoint_dir))[-1]
            os.remove(os.path.join(self.checkpoint_dir, old_ckpt))
          self.checkpoint.save(file_prefix = self.checkpoint_dir)
          
        if self.losses.generator.epoch_loss[-1] < minimum_loss:
          if (epoch-early_stop_count) > self.early_stop_count:
            logging.warning("QUIT TRAINING - Early stopping count reached. %d", self.early_stop_count)
            break
          early_stop_count=idx
          minimum_loss = self.losses.generator.epoch_loss[-1]
          tf.saved_model.save(self.generator_model, self.checkpoint_best_gen)
          tf.saved_model.save(self.discriminator_model, self.checkpoint_best_dis)
          

          
  def discriminator_v1(self, image, y=None, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      if not self.y_dim:
        h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')

        return tf.nn.sigmoid(h4), h4
      else:
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        x = conv_cond_concat(image, yb)

        h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
        h0 = conv_cond_concat(h0, yb)

        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
        h1 = tf.reshape(h1, [self.batch_size, -1])      
        h1 = concat([h1, y], 1)
        
        h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
        h2 = concat([h2, y], 1)

        h3 = linear(h2, 1, 'd_h3_lin')
        
        return tf.nn.sigmoid(h3), h3
  def discriminator(self, image_dims,):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(self.df_dim, (5,5), (2,2), "same", use_bias=True, input_shape=image_dims,
                                     kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None),
                                     bias_initializer=tf.keras.initializers.Constant(value=0),))
    model.add(tf.keras.layers.LeakyReLU())
    # model.add(tf.keras.layers.Dropout(0.3)) - repo not doing this, but paper do this

    model.add(tf.keras.layers.Conv2D(self.df_dim*2, (5,5), (2,2,), "same", use_bias=True,
                                     kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None),
                                     bias_initializer=tf.keras.initializers.Constant(value=0),))
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2D(self.df_dim*4, (5,5), (2,2,), "same", use_bias=True,
                                     kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None),
                                     bias_initializer=tf.keras.initializers.Constant(value=0),))
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2D(self.df_dim*8, (5,5), (2,2,), "same", use_bias=True,
                                     kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None),
                                     bias_initializer=tf.keras.initializers.Constant(value=0),))
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1, use_bias=True, name="h4",
                                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None),
                                    bias_initializer=tf.keras.initializers.Constant(value=0),))
    h4_logits = model.get_layer()
    model.add(tf.keras.layers.Activation(tf.nn.sigmoid))
    # return model, h4_logits
    return model


  def generator_v1(self, z_data, y_data=None):
    with tf.variable_scope("generator") as scope:
      if not self.y_dim:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # project `z` and reshape
        self.z_, self.h0_w, self.h0_b = linear(
            z_data, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)

        self.h0 = tf.reshape(
            self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(self.h0))

        self.h1, self.h1_w, self.h1_b = deconv2d(
            h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
        h1 = tf.nn.relu(self.g_bn1(self.h1))

        h2, self.h2_w, self.h2_b = deconv2d(
            h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
        h2 = tf.nn.relu(self.g_bn2(h2))

        h3, self.h3_w, self.h3_b = deconv2d(
            h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
        h3 = tf.nn.relu(self.g_bn3(h3))

        h4, self.h4_w, self.h4_b = deconv2d(
            h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

        return tf.nn.tanh(h4)
      else:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_h4 = int(s_h/2), int(s_h/4)
        s_w2, s_w4 = int(s_w/2), int(s_w/4)

        # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
        yb = tf.reshape(y_data, [self.batch_size, 1, 1, self.y_dim])
        z_data = concat([z_data, y_data], 1) # LOGICAL ERROR - z_data and y_data must have same dimensions sizes except axis

        h0 = tf.nn.relu(
            self.g_bn0(linear(z_data, self.gfc_dim, 'g_h0_lin')))
        h0 = concat([h0, y_data], 1)

        h1 = tf.nn.relu(self.g_bn1(
            linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin')))
        h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])

        h1 = conv_cond_concat(h1, yb)

        h2 = tf.nn.relu(self.g_bn2(deconv2d(h1,
            [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2')))
        h2 = conv_cond_concat(h2, yb)

        return tf.nn.sigmoid(
            deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))
  def generator(self, z,):
    # if not self.y_dim:
    s_h, s_w = self.output_height, self.output_width
    s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
    s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
    s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
    s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

    model = tf.keras.Sequential()
    
    model.add(layers.Dense(self.gf_dim*8*s_h16*s_w16, use_bias=True, input_shape=z.get_shape()[1],
                           kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None),
                           bias_initializer=tf.keras.initializers.Constant(value=0),))
    model.add(layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(layers.Reshape((s_h16, s_w16, self.gf_dim * 8)))
    assert model.output_shape == (None, s_h16, s_w16, self.gf_dim * 8)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(self.gf_dim*4, (5, 5), strides=(2, 2), padding='same', use_bias=True,
                                     kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None),
                                     bias_initializer=tf.keras.initializers.Constant(value=0),))
    assert model.output_shape == (None, s_h8, s_w8, self.gf_dim*4)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(self.gf_dim*2, (5, 5), strides=(2, 2), padding='same', use_bias=True,
                                     kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None),
                                     bias_initializer=tf.keras.initializers.Constant(value=0),))
    assert model.output_shape == (None, s_h4, s_w4, self.gf_dim*2)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(self.gf_dim, (5, 5), strides=(2, 2), padding='same', use_bias=True,
                                     kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None),
                                     bias_initializer=tf.keras.initializers.Constant(value=0),))
    assert model.output_shape == (None, s_h2, s_w2, self.gf_dim)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(self.c_dim, (5, 5), strides=(2, 2), padding='same', use_bias=True, activation='tanh',
                                     kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None),
                                     bias_initializer=tf.keras.initializers.Constant(value=0),))
    assert model.output_shape == (None, s_h, s_w, self.c_dim)

    return model
    
    # else:
    #   s_h, s_w = self.output_height, self.output_width
    #   s_h2, s_h4 = int(s_h/2), int(s_h/4)
    #   s_w2, s_w4 = int(s_w/2), int(s_w/4)

    #   yb = tf.reshape(y, [self.batch_size, 1 , 1, self.y_dim])
    #   z = tf.concat([z, y], 1)

    #   model = tf.keras.Sequential()

    #   model.add(tf.keras.layers.Dense(self.gfc_dim, use_bias=True, input_shape=[None, z.get_shape()[1]]))
    #   model.add(tf.keras.layers.BatchNormalization())
    #   model.add(tf.keras.layers.LeakyReLU())
    #   model.add(tf.keras.layers.Concatenate(axis=1)[model.output, y])
    #   assert model.output_shape == (None, self.gfc_dim, y.get_shape()[1])

    #   model.add(tf.keras.layers.Dense(self.gf_dim*2*s_h4*s_w4, use_bias=True, ))
    #   model.add(tf.keras.layers.BatchNormalization())
    #   model.add(tf.keras.layers.LeakyReLU())
    #   model.add(tf.keras.layers.Reshape(s_h4, s_w4, self.gf_dim * 2))
    #   model.add(conv_cond_concat(x=model.output, y=yb))
    #   assert model.output_shape == (None, s_h4, s_w4, self.gf_dim * 2)

    #   model.add(tf.keras.layers.Conv2DTranspose(self.gf_dim * 2, (5, 5), strides=(2, 2), padding='same', use_bias=True))
    #   model.add(tf.keras.layers.BatchNormalization())
    #   model.add(tf.keras.layers.LeakyReLU())
    #   model.add(conv_cond_concat(model.output, yb))
    #   assert model.output_shape == (None, s_h2, s_w2, self.gf_dim*2)

    #   model.add(tf.keras.layers.Conv2DTranspose(self.c_dim, (5, 5), strides=(2, 2), padding='same', use_bias=True, activation="sigmoid"))
    #   assert model.output==(None, s_h, s_w, self.c_dim)
    #   return model


  def load_builtin_dataset(self):
    logging.info("Loading built-in dataset, input_height and input_width will be reset")
    (train_images, _), (_, _) = getattr(tf.keras.datasets, self.dataset_name).load_data()
    self.input_height = self.input_width = train_images[0].shape[0]
    if train_images[0].shape[-1] == 3: # either shape will be HxW or HxWx3
      self.c_dim = 3
    else:
      self.c_dim = 1
    
    train_images = train_images.reshape(train_images.shape[0], self.input_width, self.input_height, self.c_dim).astype('float32')
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    self.buffer_size = train_images.shape[0]
    test_dataset = np.ones(train_dataset.shape[0])
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(self.buffer_size).batch(self.batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_dataset).shuffle(self.buffer_size).batch(self.batch_size)
    return train_dataset, test_dataset
  def load_custom_dataset(self):
    data_path = os.path.join(self.data_dir, self.dataset_name, self.input_fname_pattern)
    logging.info("loading custom data from %s." % data_path)
    if self.c_dim is None:
      raise ValueError("with custom data, c-dim argument is required. 1 if your data is grayscale or 3 if your dataset is RGB images")
    logging.warning("Please make sure all images are either RGB (3 channels) or grayscale (1 channels). Got argument 'c_dim'=%d" % self.c_dim)
    path_to_images = glob(data_path)
    if len(path_to_images) == 0: raise Exception("[!] No data found in '" + data_path + "'")
    if len(path_to_images) < self.batch_size: raise Exception("[!] Entire dataset size is less than the configured batch_size")
      
    if self.c_dim==1:
      data_X_np = np.stack([transform(np.array(Image.open(x).convert('L')),\
                                        self.input_height, self.input_width,\
                                        self.output_height, self.output_width, self.crop)\
                                        for x in path_to_images])
    elif self.c_dim==3:
      data_X_np = np.stack([transform(np.array(Image.open(x).convert('RGB')),\
                                        self.input_height, self.input_width,\
                                        self.output_height, self.output_width, self.crop)\
                                        for x in path_to_images])
    else:
      raise Exception("[!] Unknown color dimension. Got argument 'c_dim'=%d" % self.c_dim)
    
    self.buffer_size = data_X_np.shape[0]
    train_dataset_y = tf.data.Dataset.from_tensor_slices(np.ones(data_X_np.shape[0])).shuffle(self.buffer_size).batch(self.batch_size)
    train_dataset_x = tf.data.Dataset.from_tensor_slices(data_X_np).shuffle(self.buffer_size).batch(self.batch_size)
    return train_dataset_x, train_dataset_y


  def generator_loss(self, fake_output):
    total_loss = self.cross_entropy(tf.ones_like(fake_output), fake_output)
    # TODO instead of logits or complex structure, store actual value only
    self.losses.generator.running_loss.append(total_loss) 
    return total_loss
  def discriminator_loss(self, real_output, fake_output):
    real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    # TODO instead of logits or complex structure, store actual value only
    self.losses.discriminator.running_loss.append(total_loss)
    return total_loss
  

  @tf.function
  def train_step(self, images):
    noise = tf.random.normal([self.batch_size, self.z_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = self.generator_model(noise, training=True)

      real_output = self.discriminator_model(images, training=True)
      fake_output = self.discriminator_model(generated_images, training=True)

      gen_loss = self.generator_loss(fake_output)
      disc_loss = self.discriminator_loss(real_output, fake_output)

    # updating generator twice a step - as done in original repo
    gradients_of_generator = gen_tape.gradient(gen_loss, self.generator_model.trainable_variables)
    self.g_optim.apply_gradients(zip(gradients_of_generator, self.generator_model.trainable_variables))
    gradients_of_generator = gen_tape.gradient(gen_loss, self.generator_model.trainable_variables)
    self.g_optim.apply_gradients(zip(gradients_of_generator, self.generator_model.trainable_variables))

    gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator_model.trainable_variables)
    self.d_optim.apply_gradients(zip(gradients_of_discriminator, self.discriminator_model.trainable_variables))


  def generate_and_save_images(self, model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))
    if self.c_dim == 1:
      cmap_ = "gray"
    else:
      cmap_ = "RGB"
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap=cmap_)
        plt.axis('off')

    plt.savefig(os.path.join(self.sample_dir, 'image_at_epoch_{:04d}.png'.format(epoch)))
    plt.show()

    fig = plt.figure(figsize=(4, 4))
    plt.plot(self.losses.discriminator.epoch_loss, label="Discriminator Loss", linewidth=2, marker="*")
    plt.plot(self.losses.generator.epoch_loss, label="Generator Loss", linewidth=2, marker="*")
    plt.savefig(os.path.join(self.sample_dir, "losses.png"))

  # def sampler(self, z, y=None):
  #   with tf.variable_scope("generator") as scope:
  #     scope.reuse_variables()

  #     if not self.y_dim:
  #       s_h, s_w = self.output_height, self.output_width
  #       s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
  #       s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
  #       s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
  #       s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

  #       # project `z` and reshape
  #       h0 = tf.reshape(
  #           linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'),
  #           [-1, s_h16, s_w16, self.gf_dim * 8])
  #       h0 = tf.nn.relu(self.g_bn0(h0, train=False))

  #       h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
  #       h1 = tf.nn.relu(self.g_bn1(h1, train=False))

  #       h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
  #       h2 = tf.nn.relu(self.g_bn2(h2, train=False))

  #       h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
  #       h3 = tf.nn.relu(self.g_bn3(h3, train=False))

  #       h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')

  #       return tf.nn.tanh(h4)
  #     else:
  #       s_h, s_w = self.output_height, self.output_width
  #       s_h2, s_h4 = int(s_h/2), int(s_h/4)
  #       s_w2, s_w4 = int(s_w/2), int(s_w/4)

  #       # yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
  #       yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
  #       z = concat([z, y], 1)

  #       h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin'), train=False))
  #       h0 = concat([h0, y], 1)

  #       h1 = tf.nn.relu(self.g_bn1(
  #           linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin'), train=False))
  #       h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
  #       h1 = conv_cond_concat(h1, yb)

  #       h2 = tf.nn.relu(self.g_bn2(
  #           deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'), train=False))
  #       h2 = conv_cond_concat(h2, yb)

  #       return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))



  # @property
  # def model_dir(self):
  #   return "{}_{}_{}_{}".format(
  #       self.dataset_name, self.batch_size,
  #       self.output_height, self.output_width)

  # def save(self, checkpoint_dir, step, filename='model', ckpt=True, frozen=False):
  #   # model_name = "DCGAN.model"
  #   # checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

  #   filename += '.b' + str(self.batch_size)
  #   if not os.path.exists(checkpoint_dir):
  #     os.makedirs(checkpoint_dir)

  #   if ckpt:
  #     self.saver.save(self.tf_session,
  #             os.path.join(checkpoint_dir, filename),
  #             global_step=step)

  #   if frozen:
  #     tf.train.write_graph(
  #             tf.graph_util.convert_variables_to_constants(self.tf_session, self.tf_session.graph_def, ["generator_1/Tanh"]),
  #             checkpoint_dir,
  #             '{}-{:06d}_frz.pb'.format(filename, step),
  #             as_text=False)

  def load(self, checkpoint_dir):
    #import re
    print(" [*] Reading checkpoints...", checkpoint_dir)
    # checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
    # print("     ->", checkpoint_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.tf_session, os.path.join(checkpoint_dir, ckpt_name))
      #counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      counter = int(ckpt_name.split('-')[-1])
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0
