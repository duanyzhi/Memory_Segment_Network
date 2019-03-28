import os
import platform
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# Datasets
tf.app.flags.DEFINE_string('Google_train_data', '/media/dyz/Data/Google_Street/Img', "Google Train Datasets")
tf.app.flags.DEFINE_string('Google_test_data', '/media/dyz/Data/Google_Street/part10', "Google Test Datasets")
tf.app.flags.DEFINE_float('Google_mean', 130.62, "mean of img")
tf.app.flags.DEFINE_float('Google_std', 58.97, "std of img")
tf.app.flags.DEFINE_string('google_ckpt', 'data/ckpt/google/msn_google_103000.ckpt', "ckpt for Google")
tf.app.flags.DEFINE_float('img_size', [224, 224,  3], '''Input image size [height, weight, depth]''')

tf.app.flags.DEFINE_string('Oxford_train_data', '/media/dyz/Data/Oxford/memory', "Oxford Train Datasets")
tf.app.flags.DEFINE_string('Oxford_test_data', '/media/dyz/Data/Oxford/target', "Oxford Test Datasets")
tf.app.flags.DEFINE_float('oxford_memory_mean', 135.06, 'memory均值')
tf.app.flags.DEFINE_float('oxford_memory_std', 78.365, 'memory方差')
tf.app.flags.DEFINE_float('oxford_target_mean', 134.84, 'target均值')
tf.app.flags.DEFINE_float('oxford_target_std', 76.635, 'target方差')
tf.app.flags.DEFINE_string('oxford_ckpt', 'data/ckpt/oxford/msn_58000.ckpt', "ckpt for Google")

# Hyperparameter
tf.app.flags.DEFINE_float('weight_decay', 0.0005, "Weight decay, for regularization")
tf.app.flags.DEFINE_float('learning_rate', 0.001, "Learning rate")
tf.app.flags.DEFINE_float('momentum', 0.9, "Momentum")
tf.app.flags.DEFINE_float('gamma', 0.1, "Factor for reducing the learning rate")

tf.app.flags.DEFINE_integer('batch_size', 4, "Network batch size during training 只能是１")
tf.app.flags.DEFINE_integer('max_iter', 500000, "Max iteration")
tf.app.flags.DEFINE_integer('display_step', 20,
                            "Step size for reducing the learning rate, currently only support one step")
tf.app.flags.DEFINE_integer('img_step', 5, "the step for next img")
tf.app.flags.DEFINE_integer('test_img_step', 5, "the step for next img")
# tf.app.flags.DEFINE_integer('feature_size', 4096, "feature size of cnn output, default 256")
tf.app.flags.DEFINE_integer('hidden_size', 512, "hidden size of lstm")
tf.app.flags.DEFINE_integer('num_lstm', 10, 'number of lstm')
tf.app.flags.DEFINE_integer('memory_slice', 5, "search in 5 memory and find in which memory")
tf.app.flags.DEFINE_integer('Seg', 5, "each memory has N-shot but the shot is continuous  N frame in one memory")
tf.app.flags.DEFINE_integer('lstm_layer_num', 2, 'num of lstm layers')

# HMM
tf.app.flags.DEFINE_integer('N', 8, "number of location used in baysian")
tf.app.flags.DEFINE_float('sigma', 2, "sigma used in baysian")
tf.app.flags.DEFINE_integer('loc_threshold', 2, "real loc and pre loc less than loc_threshold, then pre is right")
tf.app.flags.DEFINE_integer('NNS', 20, "how many nearest neighborhood location, 40个定位点")
tf.app.flags.DEFINE_integer('maxN', 16, "number of location used in baysian")

# ------------------------------------------------------------------------------
