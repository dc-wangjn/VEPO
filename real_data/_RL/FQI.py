### Adapt from ###
from _util import *
import numpy as np
import tensorflow as tf
from sklearn.kernel_approximation import RBFSampler
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

import collections

### for FQE, success parallel, but overhead is very large [because many NN but few for each iter]

class MLPNetwork(tf.keras.Model):
    def __init__(self, num_actions, 
                 hiddens=[64,64], 
                 activation='relu', 
                 name='mlp_network', mirrored_strategy = None):
        super().__init__(name=name)
        
        self.num_actions = num_actions
        self.hiddens = hiddens
        self.activation = activation
        self.mirrored_strategy = mirrored_strategy
        # defining layers
#         with self.mirrored_strategy.scope():
        self.dense_layers = [tf.keras.layers.Dense(units=hidden, activation=activation)
                             for hidden in hiddens]
        ## why activation = None? softmax?
        self.out = tf.keras.layers.Dense(units=num_actions, activation=None)
        

    def call(self, state):
        net = tf.cast(state, tf.float64)
        for dense in self.dense_layers:
            net = dense(net)
        return self.out(net)
class KernelRBF(tf.keras.Model):
    def __init__(self, num_actions,
                 name='mlp_network'):
        super().__init__(name=name)

        self.out = tf.keras.layers.Dense(units=num_actions, activation=None)
        

    def call(self, s_feature):
        
        net = tf.cast(s_feature, tf.float64)
        return self.out(net)


class FQI(object):
    def __init__(self, seed,kernel,num_actions=4, init_states = None, 
                 hiddens=[256,256], activation='relu',
                 gamma=0.99
                 , lr=5e-4, decay_steps=100000,
                 batch_size=64, max_epoch=200, 
                 validation_split=0.2, es_patience=20
                 , max_iter=100, eps=0.001):
        
        self.kernel = kernel
        ### === network ===
        self.num_A = num_actions
        self.hiddens = hiddens
        self.activation = activation
        #self.mirrored_strategy = tf.distribute.MirroredStrategy() # devices=["/gpu:0"]
        self.init_states = init_states
        ### === optimization ===
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.validation_split = validation_split
        # discount factor
        self.gamma = gamma
        self.eps = eps
        self.max_iter = max_iter
        
        self.target_diffs = []
        self.values = []
        np.random.seed(seed)#wjn5-8modified
        tf.random.set_seed(seed)#wjn5-8modified
        
        ### model, optimizer, loss, callbacks ###
        #with self.mirrored_strategy.scope():
        if self.kernel :
            self.rbf_feature = RBFSampler(gamma=1, n_components=hiddens[0],random_state = 10)
            self.model = KernelRBF(self.num_A)
        else:
            
            self.model = MLPNetwork(self.num_A, hiddens, activation)
        self.optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.InverseTimeDecay(
                                                    lr, decay_steps=decay_steps, decay_rate=1))
        self.model.compile(loss= "mse" #'huber_loss'
                           , optimizer=self.optimizer, metrics=['mse'])

        self.callbacks = []
        """
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto',
            baseline=None, restore_best_weights=False
        """
        if validation_split > 1e-3:
            self.callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss'
                                                               , patience = es_patience)]        

        
    def train(self, trajs, idx = 0, train_freq = 100, test_freq = 10, verbose=0, nn_verbose = 0, validation_freq = 10):
        """
        form the target value (use observed i.o. the max) -> fit 
        """
#         print('model parms {}'.format(self.model.trainable_variables[0]))
        self.trajs = trajs
        states, actions, rewards, next_states = [np.array([item[i] for traj in trajs for item in traj]) for i in range(4)]
        states0 = np.array([traj[idx][0] for traj in trajs])
        
        
        
        if self.kernel :
            states = self.rbf_feature.fit_transform(states)
            next_states = self.rbf_feature.fit_transform(next_states)
            states0 = self.rbf_feature.fit_transform(states0)

        self.nn_verbose = nn_verbose
        # FQE
        old_targets = rewards / (1 - self.gamma)
        # https://keras.rstudio.com/reference/fit.html
        self.model.fit(states, old_targets, 
                       batch_size=self.batch_size, 
                       epochs=self.max_epoch, 
                       verbose= self.nn_verbose,
                       validation_split=self.validation_split,
                       validation_freq = validation_freq, 
                       callbacks=self.callbacks)
        ###############################################################
        for iteration in range(self.max_iter):
            q_next_states = self.model.predict(next_states)            
            targets = rewards + self.gamma * np.max(q_next_states, 1)
            ## model targets
            """ interesting. pay attention to below!!!
            """
            _targets = self.model.predict(states)
            _targets[range(len(actions)), actions.astype(int)] = targets
            self.model.fit(states, _targets, 
                           batch_size=self.batch_size, 
                           epochs=self.max_epoch, 
                           verbose = self.nn_verbose,
                           validation_split=self.validation_split,
                           validation_freq = validation_freq, 
                           callbacks=self.callbacks)
            
            self.values.append(self.V_func(states0))
            
            """works well. but why values become worse?
            Q: what does it mean? """
            target_diff = change_rate(old_targets, targets)
            self.target_diffs.append(target_diff)
            
            if verbose >= 2 and iteration % train_freq == 0:
                print('----- (training) iteration: {}, target_diff = {:.3f}, values = {:.3f}'.format(iteration, target_diff, self.values[-1].mean())
                    , '-----', '\n')
            if target_diff < self.eps:
                break
            if verbose >= 1 and iteration % test_freq == 0:
#                 print(self.init_states[:10])
                est_value = mean(self.init_state_value(self.init_states))
                print('----- (testing) iteration: {}, values = {:.2f} '.format(iteration, est_value), '-----')

            old_targets = targets.copy()
#         print('model parms {}'.format(self.model.trainable_variables[0]))
            
    """ self.model.predict
    Q values? or their action probabilities???????????

    """
    
    def V_func(self, states):
        if len(states.shape) == 1:
            states = np.expand_dim(states, 1)
        if self.kernel:
            states = self.rbf_feature.fit_transform(states)
        return np.amax(self.model(states), axis=1)
    
    def Q_func(self, states, actions = None):
        if len(states.shape) == 1:
            states = np.expand_dim(states, 1)
        if self.kernel:
            states = self.rbf_feature.fit_transform(states)
        if actions is not None:
            return np.squeeze(select_each_row(self.model(states), actions.astype(int)))
        else:
            return self.model(states)
    
    def A_func(self, states, actions = None):
        if len(states.shape) == 1:
            states = np.expand_dim(states, 1)
        if actions is not None:
            return np.squeeze(self.Q_func(states, actions.astype(int))) - self.V_func(states)
        else:
            return transpose(transpose(self.Q_func(states)) - self.V_func(states)) # transpose so that to subtract V from Q in batch.     
        
    def init_state_value(self, init_states = None, trajs = None, idx=0):
        """ TODO: Check definitions. 
        """
        if init_states is None:
            states = np.array([traj[idx][0] for traj in self.trajs])
        else:
            states = init_states
        return self.V_func(states) # len-n    
    """ NOTE: deterministic. for triply robust (off-policy learning)    
    """
    
    def get_A(self, states):
        if len(states.shape) == 1:
            states = np.expand_dim(states, 1)
        if self.kernel:
            states = self.rbf_feature.fit_transform(states)
        return np.argmax(self.model(states), axis = 1)


#     def sample_A(self, states):

#         A_prob = self.get_A_prob(states) # , to_numpy = to_numpy
#         try:
#             A_prob = A_prob.numpy()
#         except:
#             pass
#         sample_As = (A_prob.cumsum(1) > np.random.rand(A_prob.shape[0])[:,None]).argmax(1)
#         return sample_As
    
    def sample_A(self, states, to_numpy=True):
        return self.get_A(states)

    def get_A_prob(self, states, actions = None, multi_dim = False, to_numpy= True):
        """ probability for one action / prob matrix """
        if multi_dim and len(states) > 2:
            pre_dims = list(states.shape)
            pre_dims[-1] = self.num_A
            states = states.reshape(-1, self.S_dims)
        
        optimal_actions = self.get_A(states)
        
        probs = np.zeros((len(states), self.num_A))
        probs[range(len(states)), optimal_actions] = 1
#         probs = softmax(self.model.predict(states))
    
        if actions is None:
            if multi_dim and len(states) > 2:
                return probs.reshape(pre_dims)
            else:
                return probs
        else:
            return probs[range(len(actions)), actions]
    
def change_rate(old_targets, new_targets):
    diff = abs(new_targets-old_targets).mean() / (abs(old_targets).mean()+1e-6)
    return min(1.0, diff)


