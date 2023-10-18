import numpy as np
import tensorflow as tf
from _util import *

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        
################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################

class Policy(tf.keras.Model):
    ''' NN for our target policy. Policy(states) = action_probs
    '''
    def __init__(self, lamb=4.7 ,S_dims = 3, h_dims = 10, A_dims = 1, num_A = 5, depth = 1, init_paras = None):
        super(Policy, self).__init__()
        self.seed = 42
        self.hidden_dims = h_dims
        self.depth = depth
        self.input_dim = self.S_dims = S_dims
        self.num_A = num_A
        # self.input_shape2 = [self.hidden_dims, self.hidden_dims]

        if self.depth == 1:
            self.input_shape1 = [self.input_dim, self.hidden_dims]
            self.input_shape2 = [self.hidden_dims, num_A]
            if init_paras is not None:
                self.w1 = tf.Variable(tf.cast(init_paras["w1"], tf.float64), name = "w1")
                self.b1 = tf.Variable(tf.cast(init_paras["b1"], tf.float64), name = "b1")
                self.w2 = tf.Variable(tf.cast(init_paras["w2"], tf.float64), name = "w2")
                self.b2 = tf.Variable(tf.cast(init_paras["b2"], tf.float64), name = "b2")
            else:
                self.w1 = self.xavier_var_creator(self.input_shape1, name = "w1")
                self.b1 = tf.Variable(tf.zeros(self.input_shape1[1], tf.float64), name = "b1")
                self.w2 = self.xavier_var_creator(self.input_shape2, name = "w2")
                self.b2 = tf.Variable(tf.zeros(self.input_shape2[1], tf.float64), name = "b2")
            self.parms = [self.w1,self.b1,self.w2,self.b2]
            
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
            
            self.parms = [self.w1,self.b1,self.w2,self.b2,self.w3,self.b3]
        
        self.loglam = tf.Variable(lamb*tf.ones(shape=[1,1], dtype=tf.dtypes.float64), name = "loglam")
        
    def xavier_var_creator(self, input_shape, name = "w3"):
        xavier_stddev = 1.0 / tf.sqrt(input_shape[0] / 2.0)
        init = tf.random.normal(shape=input_shape, mean=0.0, stddev=xavier_stddev)
        init = tf.cast(init, tf.float64)
        var = tf.Variable(init, shape=tf.TensorShape(input_shape), trainable=True, name = name)
        return var
############################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
    def get_A_prob(self, states, actions = None, multi_dim = False, to_numpy = True):
        """ probability for one action / prob matrix """
        if multi_dim and len(states) > 2:
            pre_dims = list(states.shape)
            pre_dims[-1] = self.num_A
            states = states.reshape(-1, self.S_dims)
        
        z = tf.cast(states, tf.float64)
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
            
        probs = tf.nn.softmax(logit)
        
        if actions is None:
            if multi_dim and len(states) > 2:
                if to_numpy:
                    return probs.numpy().reshape(pre_dims)
                else:
                    return probs.reshape(pre_dims)
            else:
                return probs
        else:
            if to_numpy:
                return probs.numpy()[range(len(actions)), actions]
            else:
                return probs[range(len(actions)), actions]

    def get_greedy_A(self, states):
        """ most possible action"""
        return self.get_A_prob(states).numpy().argmax(-1)
    
    def sample_A(self, states, to_numpy = True):
        # do we really need seeds?
        # different states (each row) -> diff prob -> diff sampled actions
        # np.random.seed(self.sample_A_seed)
        # self.sample_A_seed += 1
        np.random.seed(self.seed)
        self.seed += 1       
        A_prob = self.get_A_prob(states) # , to_numpy = to_numpy
        try:
            A_prob = A_prob.numpy()
        except:
            pass
        sample_As = (A_prob.cumsum(1) > np.random.rand(A_prob.shape[0])[:,None]).argmax(1)
        return sample_As
#     def sample_A(self, states, to_numpy = True):
#         return self.get_greedy_A(states)
    
    
    def get_A(self, states, to_numpy = True):
        # do we really need seeds?
        # different states (each row) -> diff prob -> diff sampled actions
        # np.random.seed(self.sample_A_seed)
        # self.sample_A_seed += 1

        return self.sample_A(states)
    
    
class Kernel_Policy(tf.keras.Model):
    ''' NN for our target policy. Policy(states) = action_probs
    '''
    def __init__(self, lamb=4.7 ,S_dims = 3, h_dims = 10, A_dims = 1, num_A = 5, depth = 1, init_paras = None):
        super(Policy, self).__init__()
        
        
        
        
        self.seed = 42
        self.hidden_dims = h_dims
        self.depth = depth
        self.input_dim = self.S_dims = S_dims
        self.num_A = num_A
        # self.input_shape2 = [self.hidden_dims, self.hidden_dims]

        if self.depth == 1:
            self.input_shape1 = [self.input_dim, self.hidden_dims]
            self.input_shape2 = [self.hidden_dims, num_A]
            if init_paras is not None:
                self.w1 = tf.Variable(tf.cast(init_paras["w1"], tf.float64), name = "w1")
                self.b1 = tf.Variable(tf.cast(init_paras["b1"], tf.float64), name = "b1")
                self.w2 = tf.Variable(tf.cast(init_paras["w2"], tf.float64), name = "w2")
                self.b2 = tf.Variable(tf.cast(init_paras["b2"], tf.float64), name = "b2")
            else:
                self.w1 = self.xavier_var_creator(self.input_shape1, name = "w1")
                self.b1 = tf.Variable(tf.zeros(self.input_shape1[1], tf.float64), name = "b1")
                self.w2 = self.xavier_var_creator(self.input_shape2, name = "w2")
                self.b2 = tf.Variable(tf.zeros(self.input_shape2[1], tf.float64), name = "b2")
            self.parms = [self.w1,self.b1,self.w2,self.b2]
            
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
            
            self.parms = [self.w1,self.b1,self.w2,self.b2,self.w3,self.b3]
        
        self.loglam = tf.Variable(lamb*tf.ones(shape=[1,1], dtype=tf.dtypes.float64), name = "loglam")
        

############################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
    def get_A_prob(self, states, actions = None, multi_dim = False, to_numpy = True):
        """ probability for one action / prob matrix """
        if multi_dim and len(states) > 2:
            pre_dims = list(states.shape)
            pre_dims[-1] = self.num_A
            states = states.reshape(-1, self.S_dims)
        
        z = tf.cast(states, tf.float64)
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
            
        probs = tf.nn.softmax(logit)
        
        if actions is None:
            if multi_dim and len(states) > 2:
                if to_numpy:
                    return probs.numpy().reshape(pre_dims)
                else:
                    return probs.reshape(pre_dims)
            else:
                return probs
        else:
            if to_numpy:
                return probs.numpy()[range(len(actions)), actions]
            else:
                return probs[range(len(actions)), actions]

    def get_greedy_A(self, states):
        """ most possible action"""
        return self.get_A_prob(states).numpy().argmax(-1)
    
    def sample_A(self, states, to_numpy = True):
        # do we really need seeds?
        # different states (each row) -> diff prob -> diff sampled actions
        # np.random.seed(self.sample_A_seed)
        # self.sample_A_seed += 1
        np.random.seed(self.seed)
        self.seed += 1       
        A_prob = self.get_A_prob(states) # , to_numpy = to_numpy
        try:
            A_prob = A_prob.numpy()
        except:
            pass
        sample_As = (A_prob.cumsum(1) > np.random.rand(A_prob.shape[0])[:,None]).argmax(1)
        return sample_As
#     def sample_A(self, states, to_numpy = True):
#         return self.get_greedy_A(states)
    
    
    def get_A(self, states, to_numpy = True):
        # do we really need seeds?
        # different states (each row) -> diff prob -> diff sampled actions
        # np.random.seed(self.sample_A_seed)
        # self.sample_A_seed += 1

        return self.sample_A(states)
############################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
