import tensorflow.compat.v1 as tf
import numpy as np
import gzip, pickle, csv
from matplotlib import pyplot as plt
import os, random

def get_session():
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session

def set_seed(seed_number, env = None):
    os.environ['PYTHONHASHSEED']=str(seed_number)
    random.seed(seed_number)
    np.random.seed(seed_number)
    tf.compat.v1.set_random_seed(seed_number)
    if env:
        env.seed(seed_number)
    
def gs_kerl(x, y, dim_args, sigma = 20.0):
    # x, y: 2-D tensor, each of shape (batch_size * m, action_dims)
    batch_size, ac_dim = dim_args
    xx = tf.reshape(x, [batch_size, -1, ac_dim])
    yy = tf.reshape(y, [batch_size, -1, ac_dim])
    kxy = tf.square(tf.expand_dims(xx, axis = 2) - tf.expand_dims(yy, axis = 1))
    kxy = tf.reduce_sum(tf.exp(-tf.reduce_sum(kxy, axis = -1)/(2*(sigma)**2)), axis = (1,2))
    return kxy

def lp_kerl(x, y, dim_args, sigma = 10.0):
    # x, y: 2-D tensor, each of shape (batch_size * m, action_dims)
    batch_size, ac_dim = dim_args
    xx = tf.reshape(x, [batch_size, -1, ac_dim])
    yy = tf.reshape(y, [batch_size, -1, ac_dim])
    kxy = tf.abs(tf.expand_dims(xx, axis = 2) - tf.expand_dims(yy, axis = 1))
    kxy = tf.reduce_sum(tf.exp(-tf.reduce_sum(kxy, axis = -1)/(sigma)), axis = (1,2))
    return kxy

def mm_distance(x,y, num_sample, dim_args, flags = None):
    # x, y: 2-D tensor, each of shape (batch_size * m, action_dims)
    sigma = flags.sigma
    n, m = num_sample
    
    if flags.kernel == 'lp':
        mmd = lp_kerl(x, x, dim_args, sigma)/(n**2) - 2*lp_kerl(x, y, dim_args, sigma)/(n*m) + lp_kerl(y, y, dim_args, sigma)/(m**2)
    else:
        mmd = gs_kerl(x, x, dim_args, sigma)/(n**2) - 2*gs_kerl(x, y, dim_args, sigma)/(n*m) + gs_kerl(y, y, dim_args, sigma)/(m**2)
    return mmd
    
class replay_buffer(object):
    
    def __init__(self, batch_size=256):
        self.batch_size = batch_size
        
    def load_data(self, data_path):
        with gzip.open(data_path, 'rb') as f:
            replay_buffer = pickle.load(f)
        ## modified order to be consistent with our data - SARS
        self.state = np.concatenate(replay_buffer[:,0]).ravel().reshape(replay_buffer.shape[0],-1).astype('float32')
        self.next_state = np.concatenate(replay_buffer[:,3]).ravel().reshape(replay_buffer.shape[0],-1).astype('float32')
        self.action = np.concatenate(replay_buffer[:,1]).ravel().reshape(replay_buffer.shape[0],-1).astype('float32')
        self.reward = replay_buffer[:,2].astype('float32')
        # self.done = replay_buffer[:,4].astype('float32')
        replay_buffer = 0
        
    def sample(self, batch_size):
        idx = np.random.randint(self.state.shape[0],size=batch_size)
        s, ns, a, r = self.state[idx], self.next_state[idx], self.action[idx], self.reward[idx]# , d, self.done[idx]
        return s, ns, a, r#, d
    
class progress_stats(object):
    
    def __init__(self, logdir):
        self.iters = []
        self.vae_loss = []
        self.crt_loss = []
        self.act_loss = []
        self.dual_loss = []
        self.mmd = []
        self.eval_reward = []
        self.eval_reward_std = []
        self.logdir = logdir
        
    def add(self, iters, vae_loss, crt_loss, act_loss, dual_loss, mmd, eval_rew = None, eval_rew_std = None):
        self.iters.append(iters)
        self.vae_loss.append(vae_loss)
        self.crt_loss.append(crt_loss)
        self.act_loss.append(act_loss)
        self.dual_loss.append(dual_loss)
        self.mmd.append(mmd)
        self.eval_reward.append(eval_rew)
        self.eval_reward_std.append(eval_rew_std)
        
    def save_csv(self):
        with open(self.logdir + 'progress.csv', 'w') as f:
            f.write("iteration, vae_loss, crt_loss, act_loss, dual_loss, mmd, eval_reward, eval_reward_std\n")
            writer = csv.writer(f, delimiter=',')
            writer.writerows(zip(self.iters, self.vae_loss, self.crt_loss, self.act_loss, self.dual_loss, 
                                 self.mmd, self.eval_reward, self.eval_reward_std))
            
            
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
        ind = np.random.randint(0, self.storage['observations'].shape[0], size=batch_size)
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
