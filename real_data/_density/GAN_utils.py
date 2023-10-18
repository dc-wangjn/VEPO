
#############################################################################################################################
# import ccle_data
# import cit_gan
# import gan_utils

import numpy as np
import tensorflow as tf
# import tensorflow_probability as tfp
import os
import time
import math
from datetime import datetime
import logging
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from scipy.stats import wilcoxon
import random
from scipy import stats
from collections import defaultdict
import warnings
from scipy.stats import rankdata
import xlwt
from tempfile import TemporaryFile
import scipy
import pandas as pd
# import ccle_data

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
             tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

#############################################################################################################################

from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from scipy.stats import wilcoxon
import random
from scipy import stats
from collections import defaultdict
import warnings

from scipy.stats import rankdata
import xlwt
from tempfile import TemporaryFile
import scipy
import pandas as pd

tf.random.set_seed(42)
np.random.seed(42)

logging.getLogger('tensorflow').disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.keras.backend.set_floatx('float32')
total_start_time = time.time()

#############################################################################################################################
# print("TensorFlow version:", tf.__version__)

#############################################################################################################################
# Utilites related to Sinkhorn computations and training for TensorFlow 2.0

def rescale_with_quantile(x, quantile):
    
    upper = np.percentile(x, quantile, axis = 0)
    lower = np.percentile(x, 100 - quantile, axis = 0)
    
    clipped_x = np.array([np.clip(x[:,i], lower[i], upper[i]) 
                            for i in range(x.shape[1])]).transpose()
    
    
    

    x_c = (clipped_x - lower)/(upper - lower + 0.00001)
    return x_c, [lower,upper]

def cost_xy(x, y, scaling_coef):
    '''
    L2 distance between vectors, using expanding and hence is more memory intensive
    :param x: x is tensor of shape [batch_size, x dims]
    :param y: y is tensor of shape [batch_size, y dims]
    :param scaling_coef: a scaling coefficient for distance between x and y
    :return: cost matrix: a matrix of size [batch_size, batch_size] where
    '''
    x = tf.expand_dims(x, 1)
    y = tf.expand_dims(y, 0)
    return tf.reduce_sum((x - y)**2, -1) * scaling_coef


def benchmark_sinkhorn(x, y, scaling_coef, epsilon=1.0, L=10):
    '''
    :param x: a tensor of shape [batch_size, sequence length]
    :param y: a tensor of shape [batch_size, sequence length]
    :param scaling_coef: a scaling coefficient for squared distance between x and y
    :param epsilon: (float) entropic regularity constant
    :param L: (int) number of iterations
    :return: V: (float) value of regularized optimal transport
    '''
    n_data = x.shape[0]
    # Note that batch size of x can be different from batch size of y
    m = 1.0 / tf.cast(n_data, tf.float64) * tf.ones(n_data, dtype=tf.float64)
    n = 1.0 / tf.cast(n_data, tf.float64) * tf.ones(n_data, dtype=tf.float64)
    m = tf.expand_dims(m, axis=1)
    n = tf.expand_dims(n, axis=1)

    c_xy = cost_xy(x, y, scaling_coef)  # shape: [batch_size, batch_size]

    k = tf.exp(-c_xy / epsilon) + 1e-09  # add 1e-09 to prevent numerical issues
    k_t = tf.transpose(k)

    a = tf.expand_dims(tf.ones(n_data, dtype=tf.float64), axis=1)
    b = tf.expand_dims(tf.ones(n_data, dtype=tf.float64), axis=1)

    for i in range(L):
        b = m / tf.matmul(k_t, a)  # shape: [m,]
        a = n / tf.matmul(k, b)  # shape: [m,]

    return tf.reduce_sum(a * k * tf.reshape(b, (1, -1)) * c_xy)

############################################################################################################################################
def benchmark_loss(x, y, scaling_coef, sinkhorn_eps, sinkhorn_l, xp=None, yp=None):
    '''
    :param x: real data of shape [batch size, sequence length]
    :param y: fake data of shape [batch size, sequence length]
    :param scaling_coef: a scaling coefficient
    :param sinkhorn_eps: Sinkhorn parameter - epsilon
    :param sinkhorn_l: Sinkhorn parameter - the number of iterations
    :return: final Sinkhorn loss(and several values for monitoring the training process)
    '''
    if yp is None:
        yp = y
    if xp is None:
        xp = x
    x = tf.reshape(x, [x.shape[0], -1])
    y = tf.reshape(y, [y.shape[0], -1])
    xp = tf.reshape(xp, [xp.shape[0], -1])
    yp = tf.reshape(yp, [yp.shape[0], -1])
    loss_xy = benchmark_sinkhorn(x, y, scaling_coef, sinkhorn_eps, sinkhorn_l)
    loss_xx = benchmark_sinkhorn(x, xp, scaling_coef, sinkhorn_eps, sinkhorn_l)
    loss_yy = benchmark_sinkhorn(y, yp, scaling_coef, sinkhorn_eps, sinkhorn_l)

    loss = loss_xy - 0.5 * loss_xx - 0.5 * loss_yy

    return loss

############################################################################################################################################
############################################################################################################################################

class WGanGenerator(tf.keras.Model):
    '''
    class for WGAN generator
    Args:
        inputs, noise and confounding factor [v, z], of shape [batch size, z_dims + v_dims]
    return:
       fake samples of shape [batch size, x_dims]
    '''
    def __init__(self, z_dims, h_dims, v_dims, x_dims, batch_size):
        super(WGanGenerator, self).__init__()
        self.hidden_dims = h_dims
        self.batch_size = batch_size

        self.input_dim = z_dims + v_dims
        self.input_shape1 = [self.input_dim, self.hidden_dims]
        self.input_shape2 = [self.hidden_dims, self.hidden_dims]
        self.input_shape3 = [self.hidden_dims, x_dims]

        self.w1 = self.xavier_var_creator(self.input_shape1)
        self.b1 = tf.Variable(tf.zeros(self.input_shape1[1], tf.float64))

        self.w2 = self.xavier_var_creator(self.input_shape2)
        self.b2 = tf.Variable(tf.zeros(self.input_shape2[1], tf.float64))

        self.w3 = self.xavier_var_creator(self.input_shape3)
        self.b3 = tf.Variable(tf.zeros(self.input_shape3[1], tf.float64))

    def xavier_var_creator(self, input_shape):
        xavier_stddev = 1.0 / tf.sqrt(input_shape[0] / 2.0)
        init = tf.random.normal(shape=input_shape, mean=0.0, stddev=xavier_stddev)
        init = tf.cast(init, tf.float64)
        var = tf.Variable(init, shape=tf.TensorShape(input_shape), trainable=True)
        return var

    def call(self, inputs, training=None, mask=None):
        # inputs are concatenations of z and v
        z = tf.reshape(tensor=inputs, shape=[-1, self.input_dim])
#         print('z shape is {}'.format(z.shape))
        h1 = tf.nn.relu(tf.matmul(z, self.w1) + self.b1)
        h2 = tf.nn.relu(tf.matmul(h1, self.w2) + self.b2)
#         print('h2 shape is {}'.format(h2.shape))
#         print('w3 shape is {}'.format(self.w3.shape))
        out = tf.math.sigmoid(tf.matmul(h2, self.w3) + self.b3)
#         print('out shape is {}'.format(out.shape))
#         out = tf.nn.relu(tf.matmul(h2, self.w3) + self.b3)
        return out


class WGanDiscriminator(tf.keras.Model):
    '''
    class for WGAN discriminator
    Args:
        inputss: real and fake samples of shape [batch size, x_dims]
    return:
       features f_x of shape [batch size, features]
    '''
    def __init__(self, z_dims, h_dims, x_dims, batch_size):
        super(WGanDiscriminator, self).__init__()
        self.hidden_dims = h_dims
        self.batch_size = batch_size

        self.input_dim = z_dims + x_dims
        self.input_shape1 = [self.input_dim, self.hidden_dims]
        self.input_shape2 = [self.hidden_dims, self.hidden_dims]
        """pay attention to the dimension 1 here: 1 -> x_dims
        WHY NO PROBLEM ?"""
        self.input_shape3 = [self.hidden_dims, 1]

        self.w1 = self.xavier_var_creator(self.input_shape1)
        self.b1 = tf.Variable(tf.zeros(self.input_shape1[1], tf.float64))

        self.w2 = self.xavier_var_creator(self.input_shape2)
        self.b2 = tf.Variable(tf.zeros(self.input_shape2[1], tf.float64))

        self.w3 = self.xavier_var_creator(self.input_shape3)
        self.b3 = tf.Variable(tf.zeros(self.input_shape3[1], tf.float64))

    def xavier_var_creator(self, input_shape):
        xavier_stddev = 1.0 / tf.sqrt(input_shape[0] / 2.0)
        init = tf.random.normal(shape=input_shape, mean=0.0, stddev=xavier_stddev)
        init = tf.cast(init, tf.float64)
        var = tf.Variable(init, shape=tf.TensorShape(input_shape), trainable=True)
        return var

    def call(self, inputs, training=None, mask=None):
        # inputs are concatenations of z and v
        z = tf.reshape(tensor=inputs, shape=[self.batch_size, -1])
        z = tf.cast(z, tf.float64)
        h1 = tf.nn.relu(tf.matmul(z, self.w1) + self.b1)
        # h2 = tf.nn.sigmoid(tf.matmul(h1, self.w2) + self.b2)
        # out = tf.nn.sigmoid(tf.matmul(h1, self.w3) + self.b3)
        out = tf.matmul(h1, self.w3) + self.b3
        return out
############################################################################################################################################

# class MINEDiscriminator(tf.keras.layers.Layer):
#     '''
#     class for MINE discriminator for benchmark GCIT
#     '''

#     def __init__(self, in_dims, output_activation='linear'):
#         super(MINEDiscriminator, self).__init__()
#         self.output_activation = output_activation
#         self.input_dim = in_dims

#         self.w1a = self.xavier_var_creator()
#         self.w1b = self.xavier_var_creator()
#         self.b1 = tf.Variable(tf.zeros([self.input_dim, ], tf.float64))

#         self.w2a = self.xavier_var_creator()
#         self.w2b = self.xavier_var_creator()
#         self.b2 = tf.Variable(tf.zeros([self.input_dim, ], tf.float64))

#         self.w3 = self.xavier_var_creator()
#         self.b3 = tf.Variable(tf.zeros([self.input_dim, ], tf.float64))

#     def xavier_var_creator(self):
#         xavier_stddev = 1.0 / tf.sqrt(self.input_dim / 2.0)
#         init = tf.random.normal(shape=[self.input_dim, ], mean=0.0, stddev=xavier_stddev)
#         init = tf.cast(init, tf.float64)
#         var = tf.Variable(init, shape=tf.TensorShape(self.input_dim, ), trainable=True)
#         return var

#     def mine_layer(self, x, x_hat, wa, wb, b):
#         return tf.math.tanh(wa * x + wb * x_hat + b)

#     def call(self, x, x_hat):
#         h1 = self.mine_layer(x, x_hat, self.w1a, self.w1b, self.b1)
#         h2 = self.mine_layer(x, x_hat, self.w2a, self.w2b, self.b2)
#         out = self.w3 * (h1 + h2) + self.b3
#         return out, tf.exp(out)

############################################################################################################################################
