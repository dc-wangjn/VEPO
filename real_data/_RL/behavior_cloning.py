## estimate the behaviour policy ## 

import random
import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################

class BehaviorCloning:
    def __init__(self, num_actions=4, S_dims = 15, depth = 1,
                 hidden_dims = 32, activation='relu',
                 lr=5e-4, decay_steps=100000,
                 batch_size=64, max_epoch=200, 
                 validation_split=0.2, patience=20, 
                 verbose=0):
        ### === network ===
        self.num_actions = num_actions
        self.activation = activation
        self.sample_A_seed = 42
        ### === optimization ===
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.validation_split = validation_split
        self.patience = patience
        self.verbose = verbose
        
        # model, optimizer, loss, callbacks
        """ initialize the model """
        self.model = Model_Behav(num_A = num_actions, S_dims = S_dims, hidden_dims = hidden_dims, activation = activation)
        
        self.optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.InverseTimeDecay(
                         lr, decay_steps=decay_steps, decay_rate=1))
        
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        
        self.callbacks = []
        if self.validation_split > 1e-3:
            self.callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)]
        
    def train(self, states, actions):
        """ tf.keras.Model """
        self.history = self.model.fit(states, actions, # state -> action
                                      batch_size=self.batch_size, 
                                      epochs=self.max_epoch, 
                                      verbose=self.verbose,
                                      validation_split=self.validation_split,
                                      callbacks=self.callbacks)
############################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
    def get_A_prob(self, states, actions = [None], to_numpy = True):
        """
        probability for one action / prob matrix        
        sometimes prediction and sometimes training
        """
        # states shape: (batch_size, ...), actions shape: (batch_size,)
        logits = self.model(states)
        """https://stackoverflow.com/questions/55308425/difference-between-modelx-and-model-predictx-in-tensorflow
        """
        elogits = np.exp(logits)
        softmax = elogits / elogits.sum(axis=-1, keepdims=True)
        if len(actions) > 1:
            probabilities = np.squeeze(softmax[range(len(actions)), actions.astype(int)])
        else:
            probabilities = softmax
        return probabilities
        
    def get_greedy_A(self, states):
        """ most possible action
        """
        return np.argmax(self.model(states), axis=1)
    
    def sample_A(self, states):
        # do we really need seeds?
        # different states (each row) -> diff prob -> diff sampled actions
        np.random.seed(self.sample_A_seed)
        self.sample_A_seed += 1
        """ # logits? probs? ???"""
        logits = self.model(states)
        elogits = np.exp(logits)
        probs = elogits / elogits.sum(axis=-1, keepdims=True)
        sample_As = (probs.cumsum(1) > np.random.rand(probs.shape[0])[:,None]).argmax(1)
        return sample_As
    def get_A(self, states):
        # do we really need seeds?
        # different states (each row) -> diff prob -> diff sampled actions
        np.random.seed(self.sample_A_seed)
        self.sample_A_seed += 1
        """ # logits? probs? ???"""
        logits = self.model(states)
        elogits = np.exp(logits)
        probs = elogits / elogits.sum(axis=-1, keepdims=True)
        sample_As = (probs.cumsum(1) > np.random.rand(probs.shape[0])[:,None]).argmax(1)
        return self.get_greedy_A(states)
################################################################################################################################################################################################################################################################################################################################################################################################################################################

import collections

class Model_Behav(tf.keras.Model):
    def __init__(self, num_A = 5, S_dims = 15, depth = 1
                 , hidden_dims = 32,  
                 activation='relu', 
                 name='mlp_network'):
        super().__init__(name=name)
        self.input_dim = self.S_dims = S_dims
        self.num_A = num_A
        self.depth = depth
        self.hidden_dims = hidden_dims
        self.activation = activation
        # defining layers
        ########################################################
        """ we actually only have two layer here """
        
        if self.depth == 1:
            self.input_shape1 = [self.input_dim, self.hidden_dims]
            self.w1 = self.xavier_var_creator(self.input_shape1, name = "w1")
            self.b1 = tf.Variable(tf.zeros(self.input_shape1[1], tf.float64), name = "b1")
            
            self.input_shape2 = [self.hidden_dims, num_A]
            self.w2 = self.xavier_var_creator(self.input_shape2, name = "w2")
            self.b2 = tf.Variable(tf.zeros(self.input_shape2[1], tf.float64), name = "b2")
        elif self.depth == 2:
            self.input_shape1 = [self.input_dim, self.hidden_dims]
            self.w1 = self.xavier_var_creator(self.input_shape1, name = "w1")
            self.b1 = tf.Variable(tf.zeros(self.input_shape1[1], tf.float64), name = "b1")
            
            self.input_shape2 = [self.hidden_dims, self.hidden_dims]
            self.w2 = self.xavier_var_creator(self.input_shape2, name = "w2")
            self.b2 = tf.Variable(tf.zeros(self.input_shape2[1], tf.float64), name = "b2")

            self.input_shape3 = [self.hidden_dims, num_A]
            self.w3 = self.xavier_var_creator(self.input_shape3, name = "w3")
            self.b3 = tf.Variable(tf.zeros(self.input_shape3[1], tf.float64), name = "b3")
        
    def xavier_var_creator(self, input_shape, name = "w3"):
        xavier_stddev = 1.0 / tf.sqrt(input_shape[0] / 2.0)
        init = tf.random.normal(shape=input_shape, mean=0.0, stddev=xavier_stddev)
        init = tf.cast(init, tf.float64)
        var = tf.Variable(init, shape=tf.TensorShape(input_shape), trainable=True, name = name)
        return var
    def call(self, state):
        # MLPNetwork(); predict()
        z = tf.cast(state, tf.float64)
        """ can specify sparse here
        a_is_sparse = True
        https://www.tensorflow.org/api_docs/python/tf/linalg/matmul
        """
        if self.depth == 1:
            h1 = tf.nn.relu(tf.matmul(z, self.w1) + self.b1)
            logit = tf.matmul(h1, self.w2) + self.b2
        elif self.depth == 2:
            h1 = tf.nn.relu(tf.matmul(z, self.w1) + self.b1)
            h2 = tf.nn.relu(tf.matmul(h1, self.w2) + self.b2)
            logit = tf.matmul(h2, self.w3) + self.b3
        return logit
