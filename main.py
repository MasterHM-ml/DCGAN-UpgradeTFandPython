import argparse
import json
import logging
import os
import time
from glob import glob

from model import DCGAN
from utils import visualize, expand_path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s.%(msecs)03d [%(filename)s:%(lineno)d] %(levelname)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", )

parser = argparse.ArgumentParser(description="Train, Test or Infer DCGAN")

parser.add_argument("--epoch", type=int, default=200, help="Epoch to train [200]")
parser.add_argument("--g-learning-rate", type=float, default=0.001, help="Learning rate of for adam [0.0002]")
parser.add_argument("--d-learning-rate", type=float, default=0.0005, help="Learning rate of for adam [0.0002]")
parser.add_argument("--beta1", type=float, default=0.5, help="Momentum term of adam [0.5]")
parser.add_argument("--batch-size", type=int, default=64, help="The size of batch images [64]")
parser.add_argument("--c-dim", type=int, default=None,
                    help="The channel of images to use. 3 for RGB and 1 for grayscale")
parser.add_argument("--input-height", type=int, default=96,
                    help="The size of image to use (will be center cropped). [96]")
parser.add_argument("--input-width", type=int, default=None,
                    help="The size of image to use (will be center cropped). If None, same value as input_height [None]")
parser.add_argument("--output-height", type=int, default=None, help="The size of the output images to produce [None]")
parser.add_argument("--output-width", type=int, default=None,
                    help="The size of the output images to produce. If None, same value as output_height [None]")
parser.add_argument("--dataset", type=str, default="celeba", help="The name of dataset [celebA, landscape]")
parser.add_argument("--input-fname-pattern", type=str, default="*.jpg",
                    help="Glob pattern of filename of input images [*]")
parser.add_argument("--data-dir", type=str, default="./data", help="path to datasets [data]")
parser.add_argument("--out-dir", type=str, default="./output", help="Directory for saving outputs in working "
                                                                    "directory. [output]")
parser.add_argument("--checkpoint-dir", type=str, default="checkpoint",
                    help="If training or retraining; it is a folder (under out-dir/dataset+current date/) to save checkpoints e.g. "
                         "./output/celeba_2023-07-04/06:12:11/ **checkpoint-dir here** and if testing; "
                         "this path will be used independently of out-dir, then this should be a complete path "
                         "e.g. [previously_trained_model/checkpoint]")
parser.add_argument("--checkpoint-prefix", type=str, default="checkpoint",
                    help="title of model checkpoints files (under checkpoint_dir/**checkpoint-prefix here**) to save/load checkpoints e.g. "
                         "[./out/celeba_2023-07-04/06:12:11/checkpoint/checkpoint ... your model here]")
parser.add_argument("--sample-dir", type=str, default="samples",
                    help="Folder (under ./out/celeba_2023-07-04/06:12:11/ here) to save samples [./output/celeba_2023-07-04/06:12:11/samples here]")
parser.add_argument("--train", type=bool, default=False, help="True for training, False for testing [False]")
parser.add_argument("--retrain", type=bool, default=False,
                    help="True for re-training, must pass path for already saved model to resume training default: [False]")
parser.add_argument("--load-model-dir", type=str, default=None,
                    help="Folder path where model to load is being saved [None]")
parser.add_argument("--load-model-prefix", type=str, default=None,
                    help="saved model has 3 files, each start with a same prefix. Prefix under load-model-dir path [None]")
parser.add_argument("--load-best-model-only", type=bool, default=False,
                    help="If True, during testing,loading best model under checkpoint-dir/best_model/*, if False, "
                         "load latest model from checkpoint-dir [False]")
parser.add_argument("--crop", type=bool, default=False, help="True for training, False for testing [False]")
parser.add_argument("--visualize", type=bool, default=False,
                    help=" (only for RGB) True for visualizing - create a gif of generated images [False]")
parser.add_argument("--sample-freq", type=int, default=1, help="sample every this many epochs [1]")
parser.add_argument("--ckpt-freq", type=int, default=1, help="save checkpoint every this many epochs [1]")
parser.add_argument("--logging-frequency", type=int, default=1, help="print log every this many batch/iterations [1]")
parser.add_argument("--z-dim", type=int, default=100, help="dimensions of z [100]")
parser.add_argument("--generate-test-images", type=int, default=100,
                    help="Number of images to generate during test. [100]")
parser.add_argument("--images-csv-path", type=str,
                    default="/content/drive/MyDrive/Fiverr/32.DCGAN/20K_celeba_images.csv")

arguments = parser.parse_args()
logging.info(arguments)


def main(args: object):
    args.data_dir = expand_path(args.data_dir)
    args.out_dir = expand_path(args.out_dir)
    args.checkpoint_dir = expand_path(args.checkpoint_dir)
    args.sample_dir = expand_path(args.sample_dir)
    if args.output_height is None: args.output_height = args.input_height
    if args.input_width is None: args.input_width = args.input_height
    if args.output_width is None: args.output_width = args.output_height

    if args.input_height != args.input_width:
        raise ValueError("input_height and input_width must be equal, got %d height and %d width"
                         % (args.input_height, args.input_width))
    if args.output_height != args.output_width:
        raise ValueError("output_height and output_width must be equal, got %d height and %d width"
                         % (args.output_height, args.output_width))

    if args.input_height == args.input_width < 32:
        raise ValueError("input_height and input_width must be at least 32, got %d height and %d width"
                         % (args.input_height, args.input_width))
    if args.output_height == args.output_width < 32:
        raise ValueError("output_height and output_width must be at least 32, got %d height and %d width"
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

    args.out_dir = os.path.join(args.out_dir, args.dataset, str(time.strftime('%Y-%m-%d %H:%M:%S')))
    if args.train:
        args.retrain = False
        args.checkpoint_dir = os.path.join(args.out_dir, args.checkpoint_dir)
    elif args.retrain:
        args.train = False
        if (not os.path.exists(args.load_model_dir)) or (args.load_model_dir is None):
            raise Exception("Retrain mode, path to the already saved model must be passed in --load-model-dir path")
        if len(glob(os.path.join(args.load_model_dir, args.load_model_prefix + "*"))) != 3:
            raise Exception("Retrain mode, there should be exactly 3 files with %s prefix under %s. Found %d files"
                            % (args.load_model_prefix, args.load_model_dir,
                               len(glob(os.path.join(args.load_model_dir, args.load_model_prefix + "*")))))
        args.checkpoint_dir = os.path.join(args.out_dir, args.checkpoint_dir)
    else:
        args.train, args.retrain = False, False
        if not os.path.exists(args.checkpoint_dir):
            raise Exception("Inference mode: Can't find provided checkpoint directory at %s " % args.checkpoint_dir)
        if len(glob(os.path.join(args.checkpoint_dir, args.checkpoint_prefix + "*"))) != 3:
            raise Exception("Inference mode, there should be exactly 3 files with %s prefix under %s. Found %d files"
                            % (args.checkpoint_prefix, args.checkpoint_dir,
                               len(glob(os.path.join(args.checkpoint_dir, args.checkpoint_prefix + "*")))))

    args.checkpoint_prefix = os.path.join(args.checkpoint_dir, args.checkpoint_prefix)
    args.sample_dir = os.path.join(args.out_dir, args.sample_dir)

    logging.info(f"out-dir: {args.out_dir} - checkpoint-dir: {args.checkpoint_dir}"
                 f" - checkpoint-prefix: {args.checkpoint_prefix} - sample-dir: {args.sample_dir}")

    if not os.path.exists(args.out_dir): os.makedirs(args.out_dir)
    if not os.path.exists(args.checkpoint_dir): os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir): os.makedirs(args.sample_dir)

    with open(os.path.join(args.out_dir, 'arguments.json'), 'w') as f:
        arguments_dict = vars(args)
        json.dump(arguments_dict, f, indent=4, sort_keys=True, ensure_ascii=False)

    dcgan = DCGAN(
        input_width=args.input_width,
        input_height=args.input_height,
        output_width=args.output_width,
        output_height=args.output_height,
        batch_size=args.batch_size,
        c_dim=args.c_dim,
        z_dim=args.z_dim,
        train=args.train,
        dataset_name=args.dataset,
        input_fname_pattern=args.input_fname_pattern,
        crop=args.crop,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_prefix=args.checkpoint_prefix,
        load_model_dir=args.load_model_dir,
        retrain=args.retrain,
        sample_dir=args.sample_dir,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        images_csv_path=args.images_csv_path)

    logging.info("**********logging generator trainable variables**********")
    for var in dcgan.generator_model.trainable_variables:
        logging.info(f"{var.name} :: {var.shape}")
    logging.info("*******logging discriminator trainable variables**********")
    for var in dcgan.discriminator_model.trainable_variables:
        logging.info(f"{var.name} :: {var.shape}")

    if args.train:
        logging.info("***************** TRAINING *****************")
        dcgan.train(args)
    elif args.retrain:
        logging.info("***************** RE-TRAINING *****************")
        dcgan.retrain(args)
    else:
        logging.info("***************** TESTING *****************")
        dcgan.load_and_generate_images(args)

        if args.visualize:
            option = 4
            visualize(dcgan.generator_model, args, option, args.sample_dir)


if __name__ == '__main__':
    main(args=arguments)
