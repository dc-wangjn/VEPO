import tensorflow.compat.v1 as tf
import numpy as np
import tensorflow

from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib import slim as contrib_slim


class W_Network(tf.keras.Model):
    def __init__(self,hidden_dim_dr,name):
        super(W_Network, self).__init__()
#         self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim_dr
        self.activation_fn = tf.keras.activations.relu
        self.kernel_initializer = tf.keras.initializers.VarianceScaling(
        scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform')
        self.dense1 = tf.keras.layers.Dense(
        self.hidden_dim, activation=self.activation_fn,
        kernel_initializer=self.kernel_initializer, name=name)
        self.dense2 = tf.keras.layers.Dense(
        self.hidden_dim, activation=self.activation_fn,
        kernel_initializer=self.kernel_initializer, name=name)
#         self.dense3 = tf.keras.layers.Dense(
#         self.hidden_dim, activation=self.activation_fn,
#         kernel_initializer=self.kernel_initializer, name=name)
        self.dense3 = tf.keras.layers.Dense(
        1, kernel_initializer=self.kernel_initializer,
        name=name)
    def call(self,state):
        state = self.dense1(state)
        state = self.dense2(state)
        state = self.dense3(state)
        return tf.squeeze(tf.log(1+tf.exp(state)),axis=1)
class F_Network(tf.keras.Model):
    def __init__(self,hidden_dim_dr,name):
        super(F_Network, self).__init__()
#         self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim_dr
        self.activation_fn = tf.keras.activations.relu
        self.kernel_initializer = tf.keras.initializers.VarianceScaling(
        scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform')
        self.dense1 = tf.keras.layers.Dense(
        self.hidden_dim, activation=self.activation_fn,
        kernel_initializer=self.kernel_initializer, name=name)
        self.dense2 = tf.keras.layers.Dense(
        self.hidden_dim, activation=self.activation_fn,
        kernel_initializer=self.kernel_initializer, name=name)
        self.dense3 = tf.keras.layers.Dense(
        1, kernel_initializer=self.kernel_initializer,
        name=name)
    def call(self,state):
        state = self.dense1(state)
        state = self.dense2(state)
        state = self.dense3(state)
        return tf.squeeze(tf.log(1+tf.exp(state)),axis=1)
        
    
    