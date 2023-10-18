import tensorflow.compat.v1 as tf
import numpy as np

class BEAR_model(object):
    def __init__(self, dim_action = 1, num_actions = 5, latent_dim = 10):
        self.num_actions = num_actions
        self.dim_action = dim_action
        self.latent_dim = latent_dim
        
    def critic(self, obs, act, network, k = 2, reuse=False):
        # Q(s, a)
        with tf.variable_scope(network, reuse=reuse):
            with tf.variable_scope('fc1'):
                fc = tf.contrib.layers.fully_connected(tf.concat([obs, act], axis=1), self.latent_dim, activation_fn=tf.nn.relu)
                fc = tf.contrib.layers.fully_connected(fc, self.latent_dim, activation_fn=tf.nn.relu)
                fc = tf.contrib.layers.fully_connected(fc, 1, activation_fn=None)
                
            with tf.variable_scope('fc2'):
                fc2 = tf.contrib.layers.fully_connected(tf.concat([obs, act], axis=1), self.latent_dim, activation_fn=tf.nn.relu)
                fc2 = tf.contrib.layers.fully_connected(fc2, self.latent_dim, activation_fn=tf.nn.relu)
                fc2 = tf.contrib.layers.fully_connected(fc2, 1, activation_fn=None)
        # what for ?
        return tf.concat([fc, fc2], axis = 1)
    
    def actor(self, obs, network, size = 1, reuse=False):
        if size != 1:
            obs = tf.tile(obs, [size, 1])
            
        with tf.variable_scope(network, reuse=reuse):
            with tf.variable_scope('fc'):
                fc = tf.contrib.layers.fully_connected(obs, self.latent_dim, activation_fn=tf.nn.relu)
                fc = tf.contrib.layers.fully_connected(fc, self.latent_dim, activation_fn=tf.nn.relu)
                log_prob = tf.contrib.layers.fully_connected(fc, self.num_actions, activation_fn=None)
                actions = tf.random.categorical(log_prob, 1)
#                 a_prob = tf.contrib.layers.fully_connected(fc, self.num_actions, activation_fn=tf.nn.softmax)
        return actions


class VAE(object):
    """VAE Based behavior cloning also used in Fujimoto et.al. (ICML 2019)
    The generative model Gω, alongside
the value function Qθ, can be used as a policy by sampling n
actions from Gω and selecting the highest valued action according to the value estimate Qθ.

We define a differentiable constraint that approximately constrains ⇡ to ⇧, and then approximately solve the constrained optimization problem via dual gradient descent.
    """
    def __init__(self, latent_dim = 10, num_actions = 5, act_dim = 1, batch_size = 32):
        self.latent_dim = latent_dim
        self.num_actions = num_actions
        self.act_dim = act_dim
        self.batch_size = batch_size
        
    def encoder(self, obs, act, network = 'encoder', reuse = False):
        if self.continuous: 
            with tf.variable_scope(network, reuse=reuse):
                with tf.variable_scope('fc'):
                    fc = tf.contrib.layers.fully_connected(tf.concat([obs, act], axis=1), self.latent_dim, activation_fn=tf.nn.relu)
                    fc = tf.contrib.layers.fully_connected(fc, self.latent_dim, activation_fn=tf.nn.relu)
                    act_mean = tf.contrib.layers.fully_connected(fc, self.latent_dim, activation_fn=None)
                    act_logstd = tf.clip_by_value(tf.contrib.layers.fully_connected(fc, self.latent_dim, activation_fn=None), -4, 15)
                    std = tf.exp(act_logstd)

            z = act_mean + std * tf.random.normal(shape = tf.shape(std), mean= 0.0, stddev=1.0, dtype=tf.dtypes.float32)
            recon_vec, _ = self.decoder(obs = obs, z = z)
            return (recon_vec, act_mean, std)
        else:
            with tf.variable_scope(network, reuse=reuse):
                with tf.variable_scope('fc'):
                    fc = tf.contrib.layers.fully_connected(tf.concat([obs, act], axis=1), self.latent_dim, activation_fn=tf.nn.relu)
                    fc = tf.contrib.layers.fully_connected(fc, self.latent_dim, activation_fn=tf.nn.relu)
                    z = tf.contrib.layers.fully_connected(fc, self.latent_dim, activation_fn=None)
            recon_vec = self.decoder(obs = obs, z = z)
            return recon_vec
        
    def decoder(self, obs, latent_dim = 10, size = 1, z = None, network = 'decoder', reuse = False):
        if size != 1:
            obs = tf.tile(obs, [size, 1])
        
        if z is None: # ???
            mean = tf.constant(0.0, shape= [self.batch_size*size, self.latent_dim])
            std =  tf.constant(1.0, shape= [self.batch_size*size, self.latent_dim])
            z = tf.clip_by_value(mean + tf.random.normal(shape = tf.shape(std), mean= 0.0, stddev=1.0, dtype=tf.dtypes.float32), -0.5, 0.5)
        
        with tf.variable_scope(network, reuse=reuse):
            with tf.variable_scope('fc'):
                fc = tf.contrib.layers.fully_connected(tf.concat([obs, z], axis=1), latent_dim, activation_fn=tf.nn.relu)
                fc = tf.contrib.layers.fully_connected(fc, latent_dim, activation_fn=tf.nn.relu)
                log_prob = tf.contrib.layers.fully_connected(fc, self.act_dim, activation_fn=None)
                actions = tf.random.categorical(log_prob, 1)
        return actions
    
