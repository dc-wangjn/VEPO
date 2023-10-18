### Adapt from ###
from _util import *
import numpy as np
import tensorflow as tf
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
        # MLPNetwork(); predict()
        net = tf.cast(state, tf.float64)
        for dense in self.dense_layers:
            net = dense(net)
        return self.out(net)
    
########################################################################################################################################################################

class TrueQ():
    """
    learn the true Q function of a given policy
    """
    def __init__(self, trajs, gamma, policy, stochastic = True
                , num_actions = 5, hiddens=[512,512], activation='relu'):
        self.trajs = trajs
        self.N = len(self.trajs)
        self.T = len(trajs[0])
        self.gamma = gamma
        self.stochastic = stochastic
        self.policy = policy
        
        self.model = MLPNetwork(num_actions = num_actions, hiddens = hiddens, activation= activation)
        self.callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience = 1)]
        self.oganize_traj()

    def oganize_traj(self):
        def get_one_traj(j):
            SAQ = []
            for t in range(self.T):
                S = self.trajs[j][t][0]
                A = self.trajs[j][t][1]
                Rewards =  [self.trajs[j][tt][2] for tt in range(t, self.T)]
                Q = sum(Rewards[tt] * self.gamma ** tt for tt in range(len(Rewards)))
                SAQ.append([S, A, Q])
            return SAQ
        SAQ = parmap(get_one_traj, range(len(self.trajs)))
        SAQ = flatten(SAQ)
        self.states = arr([saq[0] for saq in SAQ])
        self.actions = arr([saq[1] for saq in SAQ])
        self.targets = arr([saq[2] for saq in SAQ])
    
    def train(self, gpu_number = 1, n_epoch = 500, batch_size = 1024, print_freq = 20, validation_freq = 5
                  , lr = 1e-4, decay_steps =100):
        optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.InverseTimeDecay(
                                                    lr, decay_steps = decay_steps, decay_rate = 1))
        self.model.compile(loss= "mse" #'huber_loss'
                           , optimizer=optimizer, metrics=['mse'])

        with tf.device('/gpu:' + str(gpu_number)):
            old_targets = 1e-5
            mses = zeros(n_epoch)
            es_patience = 5
            es_cnt = 0
            for i in range(n_epoch):
                idx = np.random.choice(self.N, batch_size)
                S = self.states[idx]
                full_Q = self.model.predict(S)
                Q = self.targets[idx]
                A = self.actions[idx]
                mse = tf.keras.losses.MSE(y_true = Q, y_pred = full_Q[range(batch_size), A.astype(int)])
                mses[i] = mse
                full_Q[range(batch_size), A.astype(int)] = Q
                """ better to be changed here"""
                self.model.fit(S, full_Q, 
                           batch_size=batch_size, 
                           epochs = 1, 
                           verbose= 0,
                           validation_split= .2,
                           validation_freq = validation_freq, 
                           callbacks=self.callbacks)
                if i >= 20:
                    mean_mse = mean(mses[(i - 20):i])
                    if i % print_freq == 0:
                        print("true_Q iter {} DONE! with mse = {}".format(i, mean_mse))
                        
                    rate = change_rate(old_targets, mean_mse, numpy = True)
                    if rate < 0.001:
                        es_cnt += 1
                    if es_cnt >= es_patience:
                        break
                    old_targets = mean_mse
                    
    def V_func(self, states):
        
        if len(states.shape) == 1:
            states = np.expand_dim(states, 1)
            
        if self.stochastic:
            A_probs = self.policy.get_A_prob(states, to_numpy= True)
            return np.sum(self.Q_func(states) * A_probs, 1)
            
        else:
            _actions = self.policy.get_A(states)
            return self.Q_func(states, _actions)
    
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
            states = np.array([traj[idx][0] for traj in trajs])
        else:
            states = init_states
        return self.V_func(states) # len-n
