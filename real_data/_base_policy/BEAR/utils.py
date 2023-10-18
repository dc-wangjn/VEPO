import numpy as np
import pickle
import gzip

class ReplayBuffer(object):
    def __init__(self, state_dim=10, action_dim=4):
        self.storage = dict()
        self.storage['observations'] = np.zeros((1000000, state_dim), np.float32)
        self.storage['next_observations'] = np.zeros((1000000, state_dim), np.float32)
        self.storage['actions'] = np.zeros((1000000, action_dim), np.float32)
        self.storage['rewards'] = np.zeros((1000000, 1), np.float32)
#         self.storage['terminals'] = np.zeros((1000000, 1), np.float32)
#         self.storage['bootstrap_mask'] = np.zeros((10000000, 4), np.float32)
        self.buffer_size = 1000000
        self.ctr = 0
        
    def reset(self):
        self.storage = dict()
        self.ctr = 0
        
    # Expects tuples of (state, next_state, action, reward, done)
    def add(self, data):
        self.storage['observations'][self.ctr] = data[0]
        self.storage['actions'][self.ctr] = data[1]
        self.storage['rewards'][self.ctr] = data[2]
        self.storage['next_observations'][self.ctr] = data[3]
#         self.storage['terminals'][self.ctr] = data[4]
        self.ctr += 1
        self.ctr = self.ctr % self.buffer_size

    def sample(self, batch_size, with_data_policy=False):
        ind = np.random.randint(0, self.ctr, size=batch_size)
        state, next_state, action, reward, done = [], [], [], [], []

        s = self.storage['observations'][ind]
        a = self.storage['actions'][ind]
        r = self.storage['rewards'][ind]
        s2 = self.storage['next_observations'][ind]
#         d = self.storage['terminals'][ind]
#         mask = self.storage['bootstrap_mask'][ind]

        if with_data_policy:
                data_mean = self.storage['data_policy_mean'][ind]
                data_cov = self.storage['data_policy_logvar'][ind]

                return (np.array(s), 
                        np.array(s2), 
                        np.array(a), 
                        np.array(r).reshape(-1, 1), 
                        np.array(d).reshape(-1, 1),
                        np.array(mask),
                        np.array(data_mean),
                        np.array(data_cov))

        return (np.array(s), 
                np.array(s2), 
                np.array(a), 
                np.array(r).reshape(-1, 1)
#                 , np.array(d).reshape(-1, 1),
#                 np.array(mask)
               )

    def save(self, filename):
        np.save("./TRPO_results/"+ filename +".npy", self.storage)

##################################################################################################################################
##################################################################################################################################

#     def load(self, filename, bootstrap_dim=None):
#         """Deprecated, use load_hdf5 in main.py with the D4RL environments""" 
#         with gzip.open(filename, 'rb') as f:
#                 self.storage = pickle.load(f)
        
#         sum_returns = self.storage['rewards'].sum()
#         num_traj = self.storage['terminals'].sum()
#         if num_traj == 0:
#                 num_traj = 1000
#         average_per_traj_return = sum_returns/num_traj
#         print ("Average Return: ", average_per_traj_return)
#         # import ipdb; ipdb.set_trace()
        
#         num_samples = self.storage['observations'].shape[0]
#         if bootstrap_dim is not None:
#                 self.bootstrap_dim = bootstrap_dim
#                 bootstrap_mask = np.random.binomial(n=1, size=(1, num_samples, bootstrap_dim,), p=0.8)
#                 bootstrap_mask = np.squeeze(bootstrap_mask, axis=0)
#                 self.storage['bootstrap_mask'] = bootstrap_mask[:num_samples]

                
"""
Sometimes an loss of life, done or terminal flag is used to increase the agent’s learning efficiency and, performance. This means when bootstrapping for the estimate of the expected future return, a loss of life is treated as the terminal state. The reasoning being that using the flag will allow the agent to quickly learn losing a life is to be avoided. 

We still benefit from the correct done flag being added to replay memory in the previous step.
"""