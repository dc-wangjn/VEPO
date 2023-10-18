import numpy as np
import tensorflow as tf

import collections

class MLPNetwork(tf.keras.Model):
    def __init__(self, num_actions, 
                 hiddens=[64,64], 
                 activation='relu', 
                 name='mlp_network'):
        super().__init__(name=name)
        
        self.num_actions = num_actions
        self.hiddens = hiddens
        self.activation = activation
        # defining layers
        self.dense_layers = [tf.keras.layers.Dense(units=hidden, activation=activation)
                             for hidden in hiddens]
        ## why activation = None? softmax?
        self.out = tf.keras.layers.Dense(units=num_actions, activation=None)
        
    def call(self, state):
        # MLPNetwork(); predict()
        net = tf.cast(state, tf.float32)
        for dense in self.dense_layers:
            net = dense(net)
        return self.out(net)

        
        
class WeightNetwork(tf.keras.Model):
    def __init__(self, num_actions, 
                 hiddens=[16, 16], 
                 activation='relu', 
                 name='weight'):
        super().__init__(name=name)
        
        self.num_actions = num_actions
        self.hiddens = hiddens
        self.activation = activation
        # defining layers
        self.dense_layers = [tf.keras.layers.Dense(units=hidden, activation=activation)
                             for hidden in hiddens]
        self.out = tf.keras.layers.Dense(units=1, activation=None)
        
    def call(self, states, actions, future_states):
        one_hot_actions = tf.one_hot(actions, depth=self.num_actions, 
                                     on_value=1.0, off_value=0.0)
        x = tf.concat([states, one_hot_actions, future_states], axis=-1)
        for dense in self.dense_layers:
            x = dense(x)
        return self.out(x)