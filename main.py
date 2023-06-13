import logging
import os
import argparse
# import scipy.misc
import numpy as np
import json

import time
import datetime

from model import DCGAN
from utils import visualize, expand_path

import tensorflow as tf


logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s.%(msecs)03d [%(filename)s:%(lineno)d] %(levelname)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",)


parser = argparse.ArgumentParser(description="Train, Test or Infer DCGAN")

parser.add_argument("--epoch", type=int, default=200, help="Epoch to train [200]")
parser.add_argument("--learning-rate", type=float, default=0.0002, help="Learning rate of for adam [0.0002]")
parser.add_argument("--beta1", type=float, default=0.5, help="Momentum term of adam [0.5]")
# parser.add_argument("--train-size", default=np.inf, help="The size of train images [np.inf]")
parser.add_argument("--batch-size", type=int, default=64, help="The size of batch images [64]")
parser.add_argument("--input-height", type=int, default=96, help="The size of image to use (will be center cropped). [108]")
parser.add_argument("--input-width", type=int, default=None, help="The size of image to use (will be center cropped). If None, same value as input_height [None]")
parser.add_argument("--output-height", type=int, default=None, help="The size of the output images to produce [64]")
parser.add_argument("--output-width", type=int, default=None, help="The size of the output images to produce. If None, same value as output_height [None]")
parser.add_argument("--dataset", type=str, default="mnist", help="The name of dataset [celebA, mnist, lsun]")
parser.add_argument("--input-fname-pattern", type=str, default="*.jpg", help="Glob pattern of filename of input images [*]")
parser.add_argument("--data-dir", type=str, default="./data", help="path to datasets [e.g. $HOME/data]")
parser.add_argument("--out-dir", type=str, default="./out", help="Root directory for outputs [e.g. $HOME/out]")
# parser.add_argument("--out-name", type=str, default="", help="Folder (under out_root_dir) for all outputs. Generated automatically if left blank []")
parser.add_argument("--checkpoint-dir", type=str, default="checkpoint", help="Folder (under out_root_dir/dataset+current date/) to save checkpoints e.g. out/mnist_2023-06-11/ here]")
parser.add_argument("--checkpoint-prefix", type=str, default="checkpoint", help="title of model checkpoints files (under checkpoint_dir/*) to save checkpoints e.g. out/mnist_2023-06-11/checkpoint/your model here]")
parser.add_argument("--early-stop-count", type=int, default=20, help="Number of epochs to wait before training can be quit as model is not learning")
parser.add_argument("--sample-dir", type=str, default="samples", help="Folder (under out_root_dir/out_name) to save samples [samples]")
parser.add_argument("--train", type=bool, default=False, help="True for training, False for testing [False]")
parser.add_argument("--load-best-model-only", type=bool, default=False, help="If True, during testing,loading best model under checkpoint-dir/best_model/*, if False, load latest model from checkpoint-dir [True]")
parser.add_argument("--crop", type=bool, default=False, help="True for training, False for testing [False]")
parser.add_argument("--visualize", type=bool, default=False, help="True for visualizing, False for nothing [False]")
# parser.add_argument("--export", type=bool, default=False, help="True for exporting with new batch size")
# parser.add_argument("--freeze", type=bool, default=False, help="True for exporting with new batch size")
parser.add_argument("--max-to-keep", type=int, default=3, help="maximum number of checkpoints to keep")
parser.add_argument("--sample-freq", type=int, default=1, help="sample every this many epochs")
parser.add_argument("--ckpt-freq", type=int, default=1, help="save checkpoint every this many epochs")
parser.add_argument("--z-dim", type=int, default=100, help="dimensions of z")
# parser.add_argument("--z-dist", type=str, default="uniform_signed", help="'normal01' or 'uniform_unsigned' or uniform_signed")
# parser.add_argument("--G-img-sum", type=bool, default=False, help="Save generator image summaries in log")
parser.add_argument("--generate-test-images", type=int, default=100, help="Number of images to generate during test. [100]")


args = parser.parse_args()
logging.info(args)

def main(args):
  
  # expand user name and environment variables
  args.data_dir = expand_path(args.data_dir)
  args.out_dir = expand_path(args.out_dir)
  # args.out_name = expand_path(args.out_name)
  args.checkpoint_dir = expand_path(args.checkpoint_dir)
  args.sample_dir = expand_path(args.sample_dir)


  if args.output_height is None: args.output_height = args.input_height
  if args.input_width is None: args.input_width = args.input_height
  if args.output_width is None: args.output_width = args.output_height

  if args.input_height != args.input_width:
    raise ValueError("input_height and input_width must be equal, got %d height and %d width"\
                     % (args.input_height, args.input_width))
  if args.output_height != args.output_width:
    raise ValueError("output_height and output_width must be equal, got %d height and %d width"\
                     % (args.output_height, args.output_width))

  if args.input_height == args.input_width < 32:
    raise ValueError("input_height and input_width must be at least 32, got %d height and %d width"\
                     % (args.input_height, args.input_width))
  if args.output_height == args.output_width < 32:
    raise ValueError("output_height and output_width must be at least 32, got %d height and %d width"\
                     % (args.output_height, args.output_width))
  

  if (args.input_height % 32 != 0) or (args.input_width % 32 != 0):
    raise ValueError(f"the input_height and input_width must be divisible by 32,\
                     got {args.input_height}, {args.input_width} as input.\n\n If your data has irregular shapes, pass \
                      --crop True and --output_height 'a multiple of 32 e.g. 32, 64, 96, 128, ... '.")
  
  
  if (args.output_height % 32 != 0) or (args.output_width % 32 != 0):
    raise ValueError(f"the output_height and output_width must be divisible by 32,\
                     got {args.output_height}, {args.output_width} as output, while {args.input_height}\
                        and {args.input_width} as input")
  


  if (args.output_height > args.input_height) or (args.output_width > args.input_width):
    raise ValueError(f"output_height and output_width must be smaller than input_height and input_width,\
                     got {args.output_height}, {args.output_width} as output, while {args.input_height}\
                        and {args.input_width} as input")
  
  

  # output folders
  # if args.out_name == "":
  #     args.out_name = '{} - {} - {}'.format(timestamp(), args.data_dir.split('/')[-1], args.dataset) # penultimate folder of path
  #     if args.train:
  #       args.out_name += ' - x{}.z{}.{}.y{}.b{}'.format(args.input_width, args.z_dim, args.z_dist, args.output_width, args.batch_size)


  if not os.path.exists(args.out_dir): os.makedirs(args.out_dir)
  args.out_dir = os.path.join(args.out_dir, f"{args.dataset}_{str(datetime.date.today())}")
  args.checkpoint_dir = os.path.join(args.out_dir, args.checkpoint_dir)
  args.checkpoint_prefix = os.path.join(args.checkpoint_dir, args.checkpoint_prefix)
  args.sample_dir = os.path.join(args.out_dir, f"{args.sample_dir}_{str(time.strftime('%Y-%m-%d %H:%M:%S'))}")
  
  if not os.path.exists(args.out_dir): os.makedirs(args.out_dir)
  if not os.path.exists(args.checkpoint_dir): os.makedirs(args.checkpoint_dir)
  if not os.path.exists(args.sample_dir): os.makedirs(args.sample_dir)

  with open(os.path.join(args.out_dir, 'arguments.json'), 'w') as f:
    arguments_dict = vars(args)
    json.dump(arguments_dict, f, indent=4, sort_keys=True, ensure_ascii=False)
  

  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  # run_config = tf.ConfigProto()
  # run_config.gpu_options.allow_growth=True

  # with tf.Session(config=run_config) as tf_session:
  #   if args.dataset == 'mnist':
  #     dcgan = DCGAN(
  #         tf_session,
  #         input_width=args.input_width,
  #         input_height=args.input_height,
  #         output_width=args.output_width,
  #         output_height=args.output_height,
  #         batch_size=args.batch_size,
  #         sample_num=args.batch_size,
  #         y_dim=10,
  #         z_dim=args.z_dim,
  #         dataset_name=args.dataset,
  #         input_fname_pattern=args.input_fname_pattern,
  #         crop=args.crop,
  #         checkpoint_dir=args.checkpoint_dir,
  #         sample_dir=args.sample_dir,
  #         data_dir=args.data_dir,
  #         out_dir=args.out_dir,
  #         max_to_keep=args.max_to_keep)
  #   else:
  dcgan = DCGAN(
      # tf_session,
      input_width=args.input_width,
      input_height=args.input_height,
      output_width=args.output_width,
      output_height=args.output_height,
      batch_size=args.batch_size,
      sample_num=args.batch_size,
      z_dim=args.z_dim,
      train=args.train,
      dataset_name=args.dataset,
      input_fname_pattern=args.input_fname_pattern,
      crop=args.crop,
      checkpoint_dir=args.checkpoint_dir,
      checkpoint_prefix=args.checkpoint_prefix,
      early_stop_count=args.early_stop_count,
      sample_dir=args.sample_dir,
      data_dir=args.data_dir,
      out_dir=args.out_dir,
      max_to_keep=args.max_to_keep)

  logging.info("**********logging generator trainable variables**********")
  for var in dcgan.generator_model.trainable_variables:
    logging.info(f"{var.name} :: {var.shape}")
  logging.info("*******logging discriminator trainable variables**********")
  for var in dcgan.discriminator_model.trainable_variables:
    logging.info(f"{var.name} :: {var.shape}")

  if args.train:
    logging.info("***************** TRAINING *****************")
    dcgan.train(args)
  else:
    logging.info("***************** TESTING *****************")
    # load_success = 
    # if not load_success:
      # raise Exception("Checkpoint not found in " + args.checkpoint_dir)
    dcgan.load_and_generate_images(args)


    # to_json("./web/js/layers.js", [dcgan.h0_w, dcgan.h0_b, dcgan.g_bn0],
    #                 [dcgan.h1_w, dcgan.h1_b, dcgan.g_bn1],
    #                 [dcgan.h2_w, dcgan.h2_b, dcgan.g_bn2],
    #                 [dcgan.h3_w, dcgan.h3_b, dcgan.g_bn3],
    #                 [dcgan.h4_w, dcgan.h4_b, None])

    # Below is codes for visualization
      # if args.export:
      #   export_dir = os.path.join(args.checkpoint_dir, 'export_b'+str(args.batch_size))
      #   dcgan.save(export_dir, load_counter, ckpt=True, frozen=False)

      # if args.freeze:
      #   export_dir = os.path.join(args.checkpoint_dir, 'frozen_b'+str(args.batch_size))
      #   dcgan.save(export_dir, load_counter, ckpt=False, frozen=True)

    if args.visualize:
      OPTION = 4
      visualize(dcgan.generator_model, args, OPTION, args.sample_dir)

if __name__ == '__main__':
  # tf.app.run()
  main(args)
