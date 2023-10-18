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
class SimpleReplayBuffer():

    def __init__(self, trajs,size):
        self.seed = 42
        self.states = np.array([item[0] for traj in trajs for item in traj])
        self.actions = np.array([item[1] for traj in trajs for item in traj])
        self.rewards = np.array([item[2] for traj in trajs for item in traj])
        self.next_states = np.array([item[3] for traj in trajs for item in traj])
        
        self.next_idx = self.states.shape[0] + 1
        self.S_dims = self.states.shape[1]
        
        if size < self.states.shape[0]:
            self.size = size 
        else:
            print('ENLARGE THE CAPACITY OF REPLAY BUFFER! ')
            self.size = size + self.states.shape[0]
        
    def add(self, SARS):
        if self.next_idx>=len(self.states):
            self.states = np.append(self.states, SARS[0][None,:], axis = 0)
            self.next_states = np.append(self.next_states, SARS[3][None,:], axis = 0)
            self.actions = np.append(self.actions, SARS[1])
            self.rewards = np.append(self.rewards, SARS[2])
        else:
            self.states[self.next_idx] =  SARS[0]
            self.next_states[self.next_idx] = SARS[3]
            self.actions[self.next_idx] = SARS[1]
            self.rewards[self.next_idx] = SARS[2]
        self.next_idx = int((self.next_idx + 1) % self.size)
        
    def sample(self, batch_size):
        
        np.random.seed(self.seed)
        self.seed += 1
#         print(len(self.states))
        idx = np.random.choice(len(self.states), batch_size, replace = False)
        return self.states[idx], self.actions[idx], self.rewards[idx], self.next_states[idx]
    
class FQI(object):
    def __init__(self, seed,num_actions=4, init_states = None, 
                 hiddens=[256,256], activation='relu',
                 gamma=0.99
                 , lr=5e-4, decay_steps=100000,
                 batch_size=64, max_epoch=200, 
                 validation_split=0.2, es_patience=20
                 , max_iter=100, eps=0.001):
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
        self.model = MLPNetwork(self.num_A, hiddens, activation)
        self.target_model = MLPNetwork(self.num_A, hiddens, activation)
        self.optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.InverseTimeDecay(
                                                    lr, decay_steps=decay_steps, decay_rate=1))
        self.model.compile(loss= "mse" #'huber_loss'
                           , optimizer=self.optimizer, metrics=['mse'])
        self.target_model.compile(loss= "mse" #'huber_loss'
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
            
            
    def update_target(self,target, pred):
        for t,p in zip(target.trainable_variables, pred.trainable_variables):
            t.assign(p)

        
    def train(self, trajs, idx = 0, train_freq = 100, test_freq = 10, verbose=0, nn_verbose = 0, validation_freq = 10):
        """
        form the target value (use observed i.o. the max) -> fit 
        """
#         print('model parms {}'.format(self.model.trainable_variables[0]))
        self.trajs = trajs
        states, actions, rewards, next_states = [np.array([item[i] for traj in trajs for item in traj]) for i in range(4)]
        states0 = np.array([traj[idx][0] for traj in trajs])
#         print('train states is {}'.format(states[:10]))
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
        
        self.target_model.fit(states, old_targets, 
                       batch_size=self.batch_size, 
                       epochs=self.max_epoch, 
                       verbose= self.nn_verbose,
                       validation_split=self.validation_split,
                       validation_freq = validation_freq, 
                       callbacks=self.callbacks)
        ###############################################################
        for iteration in range(self.max_iter):
            if iteration%5 ==0:
                self.update_target(self.target_model,self.model)
            
            q_next_states = self.target_model.predict(next_states)            
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
#             if target_diff < self.eps:
#                 break
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
        return np.amax(self.model(states), axis=1)
    
    def Q_func(self, states, actions = None):
        if len(states.shape) == 1:
            states = np.expand_dim(states, 1)
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
        return np.argmax(self.model(states), axis = 1)

    def get_A_prob(self, states, actions = None, multi_dim = False, to_numpy= True):
        """ probability for one action / prob matrix """
        if multi_dim and len(states) > 2:
            pre_dims = list(states.shape)
            pre_dims[-1] = self.num_A
            states = states.reshape(-1, self.S_dims)
        
        optimal_actions = self.get_A(states)
        
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

        A_prob = self.get_A_prob(states) # , to_numpy = to_numpy
        try:
            A_prob = A_prob.numpy()
        except:
            pass
        sample_As = (A_prob.cumsum(1) > np.random.rand(A_prob.shape[0])[:,None]).argmax(1)
        return sample_As
    
#     def get_A(self, states, to_numpy=True):
#         return self.sample_A(states)

class vl_env():
    def __init__(self, N, T, burnin):

        self.N = N
        self.T = T
        self.num_A = 2 
        self.burnin = burnin
        self.behav = [0.5,0.5]
        
        self.curr_state = self.reset()
    
    
    def reset(self): #modified by wjn 6-3
        
        S = np.random.randn(self.N,2)
        all_S = np.zeros((self.N,self.burnin+1,2))
        all_A = np.random.choice(self.num_A,size = self.N*(self.burnin+1),p=self.behav).reshape(self.N,self.burnin+1).astype(np.float)

        for t in range(self.burnin+1):
            all_S[:,t] = S
            S = self.step(S,all_A[:,t])
        burnin_S = all_S[:,-1]

        self.curr_state = burnin_S
        
        return self.curr_state
       
    def rew(self, state, action, next_state):
        r = next_state[:,0]*2 + next_state[:,1] - 1/4*(2*action -1)
        return r
    
    def step(self,state,action):
        n = state.shape[0]
        e = np.random.randn(2*n)*0.25
#         e = np.zeros(2*n)
        s_ = np.zeros_like(state)
#         s_[:,0] = 0.75*state[:,0] * (2*action-1) + 0.25*state[:,0]*state[:,1] + e[:n]
#         s_[:,1] = 0.75*state[:,1] * (1-2*action) + 0.25*state[:,0]*state[:,1] + e[n:]
        
        s_[:,0] = 0.75*state[:,0] * (2*action-1) + 0.25*state[:,1] + e[:n]
        s_[:,1] = 0.75*state[:,1] * (1-2*action) + 0.25*state[:,0] + e[n:]
        return s_
    
    def get_env_step(self, action):
        
        state = self.curr_state
        n = state.shape[0]
        e = np.random.randn(2*n)*0.25
#         e = np.zeros(2*n)
        s_ = np.zeros_like(state)
#         s_[:,0] = 0.75*state[:,0] * (2*action-1) + 0.25*state[:,0]*state[:,1] + e[:n]
#         s_[:,1] = 0.75*state[:,1] * (1-2*action) + 0.25*state[:,0]*state[:,1] + e[n:]
        
        s_[:,0] = 0.75*state[:,0] * (2*action-1) + 0.25*state[:,1] + e[:n]
        s_[:,1] = 0.75*state[:,1] * (1-2*action) + 0.25*state[:,0] + e[n:]        
        r = self.rew(state,action,s_)
        self.curr_state = s_
        
        return s_, r


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


class QL_online():
    def __init__(self, init_trajs, epsilon=0.05, multi_N =100, T = 100,seed=42,num_actions=4, 
                 init_states = None, 
                 hiddens=[256,256], activation='relu',
                 gamma=0.99,min_q_weight=1.0
                 , lr=5e-4, decay_steps=100000,
                 batch_size=64, max_epoch=200, 
                 replay_size = 1e5,
                 validation_split=0.2, es_patience=20
                 , max_iter=100, eps=0.001,gpu_number=0):
        
        
        self.T = T
        self.env = vl_env(N = multi_N, T = T,burnin = 50)        

        ### === network ===
        self.num_A = self.env.num_A
        #self.mirrored_strategy = tf.distribute.MirroredStrategy() # devices=["/gpu:0"]
        self.init_states = init_states

        ### === optimization ===
        self.batch_size = batch_size
        self.gpu_number = gpu_number
        # discount factor
        self.gamma = gamma

        self.max_iter = max_iter
        
        self.target_diffs = []
        self.values = []
        self.seed = seed
        
        #######
        self.step = 0
        self.e = 0
        self.epsilon = epsilon
        self.disc_values = np.zeros(100000)
        self.mean_values = np.zeros(100000)
        self.recorders = {}
        self.replay_buffer = SimpleReplayBuffer(init_trajs,replay_size)
        
        ### model, optimizer, loss, callbacks ###
        #with self.mirrored_strategy.scope():
        self.model = MLPNetwork(self.num_A, hiddens, activation 
                                #, mirrored_strategy = self.mirrored_strategy
                               )
        self.optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.InverseTimeDecay(
                                                    lr, decay_steps=decay_steps, decay_rate=1))
        self.model.compile(loss= "mse"  , optimizer=self.optimizer, metrics=['mse'])

        self.callbacks = []
        """
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto',
            baseline=None, restore_best_weights=False
        """
        if validation_split > 1e-3:
            self.callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss'
                                                               , patience = es_patience)] 
            
        with tf.device('/gpu:' + str(self.gpu_number)):
            states, actions, rewards, next_states = self.replay_buffer.sample(self.batch_size)
            old_targets = rewards / (1 - self.gamma)
            self.model.fit(states, old_targets, batch_size=self.batch_size )
            
    def fit_one_step(self, print_freq = 5, verbose = 0, nn_verbose = 0):
        with tf.device('/gpu:' + str(self.gpu_number)):
            states, actions, rewards, next_states = self.replay_buffer.sample(self.batch_size)
            q_next_states = self.model.predict(next_states)
            targets = rewards + self.gamma * np.max(q_next_states, 1)
            _targets = self.model.predict(states)
            _targets[range(len(actions)), actions.astype(int)] = targets
            with tf.GradientTape() as tape:
                pred_targets = self.model(states)
                loss = tf.keras.losses.MSE(_targets, pred_targets)

            dw = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(dw, self.model.trainable_variables))
            
            if verbose >= 1 and self.step % print_freq == 0:
                print('----- FQI (training) iteration: {},  values = {:.3f}'.format(self.step,  targets.mean()) , '-----')
                
    def train_one_epoch(self):
        i = self.e
        """ not useful here """
        observation = self.env.reset()

        self.recorders[i] = {"state" : [], "action" : [], "reward" : []}
        observation = self.env.reset()
        t = 0
        while True or t <= self.T:
            if rbin(n = 1, p = self.epsilon):
                action = np.random.choice(self.num_A,self.env.N, p = np.repeat(1, self.num_A) / self.num_A)
            else:
                action = self.get_A(observation)
            observation_, reward = self.env.get_env_step(action)
            for k in range(len(observation)):
                self.replay_buffer.add([observation[k], action[k], reward[k], observation_[k]])

            self.recorders[i]["state"].append(observation)
            self.recorders[i]["action"].append(action)
            self.recorders[i]["reward"].append(reward)
            self.fit_one_step(print_freq = 5, verbose = 1, nn_verbose = 0)

            if t == self.T:
                running_rewards = np.array([r for r in self.recorders[i]["reward"]])
                mean_reward = mean(running_rewards)
                gammas = np.repeat(arr([self.gamma ** t for t in range(len(running_rewards))])[:,None],self.env.N,axis=1)
                self.disc_values[i] = np.mean(np.sum(running_rewards * gammas,axis=0))
                self.mean_values[i] = mean_reward
                if i >= 10 and i % 10 == 0:
                    if i >= 100:
                        v = np.mean(self.disc_values[(i - 100):i])
                    elif i >= 50:
                        v = np.mean(self.disc_values[(i - 50):i])
                    else:
                        v = np.mean(self.disc_values[(i - 10):i])
                    print("episode:", i, "  reward:", v)

                elif i < 10:
                    v = self.disc_values[i]
                    print("episode:", i, "  reward:", v)
                break

            observation = observation_
            t += 1
            self.step += 1
#         if i > 0 and i % save_freq == 0:
#             self.dqn.model.save_weights(self.FQI_path + "/iter" + str(i))
        self.e += 1       
            

        
    def train(self):
        
        while self.e < self.max_iter:
            self.train_one_epoch()
            

#     """
    
    def V_func(self, states):
        if len(states.shape) == 1:
            states = np.expand_dim(states, 1)
        return np.amax(self.model(states), axis=1)
    
    def Q_func(self, states, actions = None):
        if len(states.shape) == 1:
            states = np.expand_dim(states, 1)
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
        return np.argmax(self.model(states), axis = 1)

    def get_A_prob(self, states, actions = None, multi_dim = False, to_numpy= True):
        """ probability for one action / prob matrix """
        if multi_dim and len(states) > 2:
            pre_dims = list(states.shape)
            pre_dims[-1] = self.num_A
            states = states.reshape(-1, self.S_dims)
        
        optimal_actions = self.get_A(states)
        
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

        A_prob = self.get_A_prob(states) # , to_numpy = to_numpy
        try:
            A_prob = A_prob.numpy()
        except:
            pass
        sample_As = (A_prob.cumsum(1) > np.random.rand(A_prob.shape[0])[:,None]).argmax(1)
        return sample_As
    
#     def get_A(self, states, to_numpy=True):
#         return self.sample_A(states)


    
def change_rate(old_targets, new_targets):
    diff = abs(new_targets-old_targets).mean() / (abs(old_targets).mean()+1e-6)
    return min(1.0, diff)


