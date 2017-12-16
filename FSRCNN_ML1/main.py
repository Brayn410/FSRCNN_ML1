from model import FSRCNN

import numpy as np
import tensorflow as tf

import pprint
import os

PREFIX_PATH = "D:/Dropbox/Dropbox/Master Theoretische Informatik/Machine Learning 1/Python Programme ML1 Project/FSRCNN_New/"

"""
ToDo:
Upscaling by an arbitrary factor (at the moment only upscaling by factor 3 works)
It seems like that "scale_factors = [[14,20], [11, 21], [10, 24]]" changes the crop of the images (see model.py) 
    1. It seems that [u,v] for a given v, u will be calculated as follows: u = v/scaling_factor + 4      but why? All other combinations (without this relation) gets an error.
    2. If we change [11,21] to [7,9] it holds v = 9 and u = 9/3 + 4 = 7. We do not get an error for [7,9], but the resulting image is smaller than for [11,21], why?
    3. What have to be changed, that 2. will not occur?
Where does the black or white grid comes from by to less training (see result folder after test-run)
"""





flags = tf.app.flags

flags.DEFINE_string("PREFIX_PATH", PREFIX_PATH, "Path to the main.py file.")
flags.DEFINE_boolean("fast", True, "Use the fast model (FSRCNN-s) [False]")
flags.DEFINE_integer("epoch", 150, "Number of epochs [10]")
flags.DEFINE_integer("batch_size", 32768, "The size of batch images [128]")
flags.DEFINE_float("learning_rate", 0.01, "The learning rate of gradient descent algorithm [1e-3]")

# Momentum is not necessary anymore, because we use Adam-Optimizer
flags.DEFINE_float("momentum", 0.9, "The momentum value for the momentum SGD [0.9]")

flags.DEFINE_integer("seed", 123, "Random seed [123]")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color [1]")
flags.DEFINE_integer("scale", 3, "The size of scale factor for preprocessing input image [3]")
flags.DEFINE_integer("stride", 4, "The size of stride to apply to input image [4]")
flags.DEFINE_string("checkpoint_dir", PREFIX_PATH + "checkpoint/", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("output_dir", PREFIX_PATH + "result/", "Name of test output directory [result]")
flags.DEFINE_string("data_dir", PREFIX_PATH + "Train_new/", "Name of data directory to train on [FastTrain]")
flags.DEFINE_boolean("train", False, "True for training, false for testing [True]")
flags.DEFINE_integer("threads", 1, "Number of processes to pre-process data with [1]")
flags.DEFINE_boolean("params", False, "Save weight and bias parameters [False]")

FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if FLAGS.fast:
    FLAGS.checkpoint_dir = PREFIX_PATH + "checkpoint_fast/"
  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)


  with tf.Session() as sess:
    tf.set_random_seed(FLAGS.seed)
    fsrcnn = FSRCNN(sess, config=FLAGS)
    fsrcnn.run()
    
#if __name__ == '__main__':
tf.app.run()
