### Adapt from ###
from _util import *
import numpy as np
import tensorflow as tf


from sklearn.ensemble import RandomForestRegressor


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

import collections

### for FQE, success parallel, but overhead is very large [because many NN but few for each iter]


"""
batch RL
deterministic policies <- Q-based
model is the Q-network
"""

class FQI(object):
    def __init__(self, num_actions=4, init_states = None, 
                 max_depth=2,random_state=0,warm_start=True,
                 gamma=0.99 , max_iter=100, eps=0.001):

        self.num_actions = num_actions
        self.init_states = init_states
        self.max_depth = max_depth
        self.random_state = random_state
        self.warm_start = warm_start
        # discount factor
        self.gamma = gamma
        self.eps = eps
        self.max_iter = max_iter
        
        self.target_diffs = []
        self.values = []
        
        ### model, optimizer, loss, callbacks ###
        #with self.mirrored_strategy.scope():
        
#         self.optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.InverseTimeDecay(
#                                                     lr, decay_steps=decay_steps, decay_rate=1))
#         self.model.compile(loss= "mse" #'huber_loss'
#                            , optimizer=self.optimizer, metrics=['mse'])

#         self.callbacks = []
#         """
#         tf.keras.callbacks.EarlyStopping(
#             monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto',
#             baseline=None, restore_best_weights=False
#         """
#         if validation_split > 1e-3:
#             self.callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss'
#                                                                , patience = es_patience)]        

        
    def train(self, trajs, idx = 0, train_freq = 100, test_freq = 10, verbose=0):
        """
        form the target value (use observed i.o. the max) -> fit 
        """
#         time = now()
        self.trajs = trajs
        states, actions, rewards, next_states = [np.array([item[i] for traj in trajs for item in traj]) for i in range(4)]
        states0 = np.array([traj[idx][0] for traj in trajs])
#         print('init cost is {}'.format(now()-time))
        

        # FQE
        
        old_targets = rewards / (1 - self.gamma)
        old_targets = np.repeat(np.expand_dims(old_targets,1),self.num_actions,axis=1)
        # https://keras.rstudio.com/reference/fit.html
        time = now()
        model = RandomForestRegressor(
           max_depth=self.max_depth,random_state=self.random_state,
            warm_start=self.warm_start)
        model.fit(states, old_targets)
        old_targets = old_targets.mean(axis=1)
#         print('init fit cost is {}'.format(now()-time))
        ###############################################################
        for iteration in range(self.max_iter):
            q_next_states = model.predict(next_states)            
            targets = rewards + self.gamma * np.max(q_next_states, 1)
            ## model targets
            """ interesting. pay attention to below!!!
            """
            time = now()
            _targets = model.predict(states)
#             print(_targets[:10])
#             print('predict cost is {}'.format(now()-time))
            _targets[range(len(actions)), actions.astype(int)] = targets
#             print(_targets[:10])
            time = now()
            model = RandomForestRegressor(
               max_depth=self.max_depth,random_state=self.random_state,
                warm_start=self.warm_start)
            model.fit(states, _targets)
#             print('fit cost is {}'.format(now()-time))
            time = now()
#             self.values.append(self.V_func(states0))
            
            """works well. but why values become worse?
            Q: what does it mean? """
            target_diff = change_rate(old_targets, targets)
            self.target_diffs.append(target_diff)
            self.model = model  
            
            if verbose >= 2 and iteration % train_freq == 0:
                print('----- (training) iteration: {}, target_diff = {:.3f}, values = {:.3f}'.format(iteration, target_diff, self.values[-1].mean())
                    , '-----', '\n')
#             if target_diff < self.eps:
#                 break
            if verbose >= 1 and iteration % test_freq == 0:
                est_value = mean(self.init_state_value(self.init_states))
                print('----- (testing) iteration: {}, values = {:.2f} '.format(iteration, est_value), '-----')

            old_targets = targets.copy()
#             print('trifle cost is {}'.format(now()-time))
                  
    """ self.model.predict
    Q values? or their action probabilities???????????

    """
    
    def V_func(self, states):
        if len(states.shape) == 1:
            states = np.expand_dim(states, 1)
        return np.amax(self.model.predict(states), axis=1)
    
    def Q_func(self, states, actions = None):
        if len(states.shape) == 1:
            states = np.expand_dim(states, 1)
        if actions is not None:
            return np.squeeze(select_each_row(self.model.predict(states), actions.astype(int)))
        else:
            return self.model.predict(states)
    
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
        return np.argmax(self.model.predict(states), axis = 1)

    def get_A_prob(self, states, actions = None, multi_dim = False, to_numpy= True):
        """ probability for one action / prob matrix """
        if multi_dim and len(states) > 2:
            pre_dims = list(states.shape)
            pre_dims[-1] = self.num_A
            states = states.reshape(-1, self.S_dims)
        
#         optimal_actions = self.get_A(states)
        
#         probs = np.zeros((len(states), self.num_A))
#         probs[range(len(states)), optimal_actions] = 1
        
        
        probs = softmax(self.model.predict(states))
        

        if actions is None:
            if multi_dim and len(states) > 2:
                return probs.reshape(pre_dims)
            else:
                return probs
        else:
            return probs[range(len(actions)), actions]
    def sample_A(self, states):
        return self.get_A(states)


    
def change_rate(old_targets, new_targets):
    diff = abs(new_targets-old_targets).mean() / (abs(old_targets).mean()+1e-6)
    return min(1.0, diff)


