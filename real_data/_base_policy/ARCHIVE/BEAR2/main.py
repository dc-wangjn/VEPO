import os
import tensorflow as tf
from tensorflow.python.platform import flags
import numpy as np
import gym
import mujoco_py
import gzip
import pickle
import time

from bear_model import BEAR_model
from bear_qlearner import BEAR_qlearner

FLAGS = flags.FLAGS

# Game environments
flags.DEFINE_string('game', 'Walker2d', 'Mujoco env')
flags.DEFINE_string('version', 'v2', 'env version')
flags.DEFINE_integer('seed', 7701, 'seed')
flags.DEFINE_integer('it', 1000000, 'training step')
flags.DEFINE_integer('batch', 100, 'training step')
flags.DEFINE_float('lr', 1e-3, 'training step')
flags.DEFINE_integer('sample_p', 5, 'training step')
flags.DEFINE_integer('sample', 5, 'training step')
flags.DEFINE_integer('eval_freq', 1000, 'training step')
flags.DEFINE_string('exp', 'exp', 'experiment name')
flags.DEFINE_string('kernel', 'lp', 'kernel for mmd')
flags.DEFINE_float('sigma', 10.0, 'sigma for mmd')
flags.DEFINE_string('buffer', 'buffers', 'file name of buffer')

def train_bear():
    env = gym.make(FLAGS.game+'-'+FLAGS.version)
    logdir = './results/' + FLAGS.game+'-'+FLAGS.version + '_seed' + str(FLAGS.seed) + '_' + FLAGS.exp + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    flags.DEFINE_string('logdir', logdir + '/', 'logdir')
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    
    bear_model = BEAR_model(num_actions = env.action_space.shape[0], max_action = float(env.action_space.high[0]))
    bear_qlearner = BEAR_qlearner(env = env, model = bear_model, FLAGS = FLAGS,  graph_args = {'p':FLAGS.sample_p, 'm':FLAGS.sample, 'n':FLAGS.sample,
                                                                                               'lr':FLAGS.lr, 'batch_size':FLAGS.batch, 'lambda':0.75,
                                                                                               'eval_freq':FLAGS.eval_freq, 'var_lambda':0.4, 
                                                                                               'eps':0.05,'tau':5e-3, 'gamma':0.99,})
    bear_qlearner.train(iterations = FLAGS.it)

if __name__ == "__main__":
    train_bear()





