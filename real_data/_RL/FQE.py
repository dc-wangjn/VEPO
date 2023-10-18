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
from livelossplot import PlotLosses

################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
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
        self.out = tf.keras.layers.Dense(units=num_actions, activation=None) #  ## why activation = None? softmax?

    def call(self, state):
        # MLPNetwork(); predict()
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
    

class FQE(object):
    def __init__(self, seed, kernel, policy, stochastic = True, depth = 1, # policy to be evaluated
                 num_actions=4, init_states = None, 
                 hiddens=[256,256], activation='relu',
                 gamma=0.99, update_plot = True
                 , lr=5e-4, decay_steps=100000,
                 batch_size=64, max_epoch=200, 
                 validation_split=0.2, es_patience=20
                 , gpu_number = 0
                 , max_iter=100, eps=0.001):
        self.kernel = kernel
        self.policy = policy
        ### === network ===
        self.num_actions = num_actions
        self.hiddens = hiddens
        self.activation = activation
        self.stochastic = stochastic
        self.init_states = init_states
        ### === optimization ===
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.gpu_number = gpu_number
        self.validation_split = validation_split
        # discount factor
        self.gamma = gamma
        self.eps = eps
        self.max_iter = max_iter
        self.target_diffs = []
        self.values = []
        self.update_plot = update_plot
        tf.random.set_seed(seed)#wjn5-8modified
        np.random.seed(seed)#wjn5-8modified

        
#         self.mirrored_strategy = tf.distribute.MirroredStrategy() # devices=["/gpu:0"]
#         self.n_gpu = 1
#         print('Number of devices: {}'.format(self.mirrored_strategy.num_replicas_in_sync))
        ### model, optimizer, loss, callbacks ###
#         with self.mirrored_strategy.scope():
#         self.model = MLPNetwork(num_actions, hiddens, activation 
#                                #, mirrored_strategy = self.mirrored_strategy
#                                )
        if self.kernel :
            self.rbf_feature = RBFSampler(gamma=1, n_components=hiddens[0],random_state = 10)
            self.model = KernelRBF(num_actions)
        else:
            
            self.model = MLPNetwork(num_actions, hiddens, activation)
        self.optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.InverseTimeDecay(
                                                    lr, decay_steps=decay_steps, decay_rate=1))
        self.model.compile(loss= "mse" #'huber_loss'
                           , optimizer=self.optimizer, metrics=['mse'])
        
        """
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto',
            baseline=None, restore_best_weights=False
        """
        self.callbacks = []
        if validation_split > 1e-3:
            self.callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss'
                                                               , patience = es_patience)]        

        
    def train(self, trajs, idx = 0, train_freq = 100, test_freq = 10, verbose=0, nn_verbose = 0, validation_freq = 10
             ):
        """
        form the target value (use observed i.o. the max) -> fit 
        """
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
#         curr = now()
#         fit_time = 0
#         policy_time = 0
        with tf.device('/gpu:' + str(self.gpu_number)):
            self.model.fit(states, old_targets, 
                       batch_size=self.batch_size, 
                       epochs=self.max_epoch, 
                       verbose= self.nn_verbose,
                       validation_split=self.validation_split,
                       validation_freq = validation_freq, 
                       callbacks=self.callbacks)
        
#         fit_time += now() - curr; curr = now()
        ###############################################################
        test_values = []
        groups = {'Value': ['Value']}
        plotlosses = PlotLosses(groups=groups)

        with tf.device('/gpu:' + str(self.gpu_number)):
            for iteration in range(self.max_iter):
#                 print('now the {}-iter'.format(iteration))
                #############################################
                ############# Model the target ##############
                q_next_states = self.model.predict(next_states)
                if self.stochastic:
                    A_probs = self.policy.get_A_prob(next_states, to_numpy= True)
                    targets = rewards + self.gamma * np.sum(q_next_states * A_probs, 1)
                else:
                    _actions = self.policy.get_A(next_states)
                    targets = rewards + self.gamma * q_next_states[range(len(_actions)), _actions]  #*(1.0-dones)

    #             policy_time += now() - curr; curr = now()
    #             print("policy_time", policy_time)
                ###### Why we must train things in such a way?
                ###### Why not just the original Q(S, A)
                ## model targets
                _targets = self.model.predict(states)
                _targets[range(len(actions)), actions.astype(int)] = targets
                #############################################
                #############################################
                # with tf.device('/gpu:5'):
                self.model.fit(states, _targets, 
                               batch_size=self.batch_size, 
                               epochs=self.max_epoch, 
                               verbose = self.nn_verbose,
                               validation_split=self.validation_split,
                               validation_freq = validation_freq, 
                               callbacks=self.callbacks)

    #             fit_time += now() - curr; curr = now()
    #             print("fit_time", fit_time)
                ##########################################################################################
                target_diff = change_rate(old_targets, targets)
                self.target_diffs.append(target_diff)

                # self.values.append(self.V_func(states0))
                """ we can modify the stop creteria to be one w.r.t the init_states """
                if verbose >= 1 and iteration % test_freq == 0:
                    # est_value = mean(self.init_state_value(self.init_states))
                    if self.stochastic:
                        A_probs = self.policy.get_A_prob(self.init_states, to_numpy= True)
                        values = np.sum(self.Q_func(self.init_states) * A_probs, 1)
#                         print(self.init_states[55:60])
#                         print(values.shape)
#                         print(self.init_states.shape)
#                         print('value show {}'.format(values[:10]))
                    else:
                        _actions = self.policy.get_A(self.init_states)
                        values = self.Q_func(self.init_states, _actions)
#                         print(values.shape)
#                         print(self.init_states.shape)
#                         print('value show {}'.format(values[:10]))
#                         print(self.init_states[55:60])

                    est_value = np.mean(values)
                    test_values.append(est_value)
                    if self.update_plot:
                        plotlosses.update({
                            'Value': est_value
                        })
                        plotlosses.send()
                    else:
                        print("Value of FQE iter {} = {:.2f} with diff = {:.4f}".format(iteration, est_value, target_diff))

                ######## Stopping Creteria Here ########
                if target_diff < self.eps:
                    break
                old_targets = targets.copy()
            ##########################################################################################
################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
    """ model: Q-func -> V, A
    V_values = pi_value.state_value(generated_S) # len = n_states
    Q_values = pi_value.state_action_value(generated_S) # n_states * range_A
    A_values = transpose(transpose(Q_values) - V_values) # n_states (NN * TT) * range_A

    """
    
    def V_func(self, states):
        
        if len(states.shape) == 1:
            states = np.expand_dim(states, 1)
            
        if self.stochastic:
            A_probs = self.policy.get_A_prob(states, to_numpy= True)
            return np.sum(self.Q_func(states) * A_probs, 1)
            
        else:
            _actions = self.policy.get_A(states)
            return self.Q_func(states, _actions)
        
#         return np.amax(self.model.predict(states), axis=1)
    
    def Q_func(self, states, actions = None):
        if len(states.shape) == 1:
            states = np.expand_dim(states, 1)
        if self.kernel:
            states = self.rbf_feature.fit_transform(states)
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
################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
    
def change_rate(old_targets, new_targets):
    diff = abs(new_targets-old_targets).mean() / (abs(old_targets).mean()+1e-6)
    return min(1.0, diff)


