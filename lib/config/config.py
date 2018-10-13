import os
import platform
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# Datasets
tf.app.flags.DEFINE_string('Google_train_data', 'data/Google/train')
tf.app.flags.DEFINE_string('Google_test_data', 'data/Google/test')
tf.app.flags.DEFINE_float('Google_mean', 130.62, "mean of img")
tf.app.flags.DEFINE_float('Google_std', 58,97, "std of img")


# Hyperparameter
tf.app.flags.DEFINE_float('weight_decay', 0.0005, "Weight decay, for regularization")
tf.app.flags.DEFINE_float('learning_rate', 0.001, "Learning rate")
tf.app.flags.DEFINE_float('momentum', 0.9, "Momentum")
tf.app.flags.DEFINE_float('gamma', 0.1, "Factor for reducing the learning rate")

tf.app.flags.DEFINE_integer('batch_size', 16, "Network batch size during training")
tf.app.flags.DEFINE_integer('max_iter', 500000, "Max iteration")
tf.app.flags.DEFINE_integer('display_step', 20,
                            "Step size for reducing the learning rate, currently only support one step")
tf.app.flags.DEFINE_integer('img_step', 5, "the step for next img")
tf.app.flags.DEFINE_integer('test_img_step', 5, "the step for next img")

tf.app.flags.DEFINE_integer('feature_size', 512, "feature size of cnn output")
tf.app.flags.DEFINE_integer('hidden_size', 512, "hidden size of lstm")
tf.app.flags.DEFINE_integer('num_lstm', 10, 'number of lstm')
tf.app.flags.DEFINE_integer('memory_size', 5, "search in 5 memory and find in which memory")
tf.app.flags.DEFINE_integer('shot', 5, "each memory has N-shot but the shot is continuous  N frame in one memory")
tf.app.flags.DEFINE_integer('lstm_layer_num', 2, 'num of lstm layers')


tf.app.flags.DEFINE_float('img_size', [224, 224,  3], '''Input image size [height, weight, depth]''')

# ------------------------------------------------------------------------------
