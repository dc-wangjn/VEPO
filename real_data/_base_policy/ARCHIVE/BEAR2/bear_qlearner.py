import tensorflow.compat.v1 as tf
import numpy as np
import gzip
import pickle
import time
from .utils import lp_kerl, mm_distance, get_session, set_seed, replay_buffer, progress_stats
from matplotlib import pyplot as plt

class BEAR_qlearner(object):
    def __init__(self, model, FLAGS, buffer_model, buffer
                 , cont = False
                 , ob_dim = 3, ac_dim = 1, env = None
                 , graph_args = {'p':5, 'm':5, 'n':5, 'lambda':0.75, 'var_lambda':0.4, 'lr':1e-3,
                                                        'eps':0.05, 'tau':5e-3, 'gamma':0.99, 'batch_size':256, 'eval_freq':1000}):
        self.env = env
        self.model = model
        self.cont = cont
        self.ob_dim = ob_dim #env.observation_space.shape[0]
        self.ac_dim = ac_dim # env.action_space.shape[0]
        self.flags = FLAGS
        self.p = graph_args['p'] 
        self.m = graph_args['m']
        self.n = graph_args['n'] 
        self.ld = graph_args['lambda'] 
        self.var_ld = graph_args['var_lambda']
        self.learning_rate = graph_args['lr']
        self.eps = graph_args['eps']
        self.tau = graph_args['tau'] 
        self.gamma = graph_args['gamma'] 
        self.batch_size = graph_args['batch_size'] 
        self.eval_freq = graph_args['eval_freq'] 
        self.buffer_model = buffer_model
        self.buffer = buffer
        """ Q: done? """
    def define_placeholder(self):
        self.ob = tf.placeholder(shape=[None, self.ob_dim], dtype=tf.float32)
        self.ac = tf.placeholder(shape=[None, self.ac_dim], dtype=tf.float32) 
        self.rew = tf.placeholder(shape=[None], dtype=tf.float32)
        #self.done = tf.placeholder(shape=[None], dtype=tf.float32) 
        self.next_ob = tf.placeholder(shape=[None, self.ob_dim], dtype=tf.float32)
        
        self.vloss = tf.placeholder(dtype=tf.float32)
        self.closs = tf.placeholder(dtype=tf.float32)
        self.aloss = tf.placeholder(dtype=tf.float32)
        self.dloss = tf.placeholder(dtype=tf.float32)
        self.distance = tf.placeholder(dtype=tf.float32)
        
    def construct_graph(self):
        self.define_placeholder()
        
        # initialize actor and vae network (samle actions for critic and actor)

        buffer_vae_recon = self.buffer_model.encoder(self.ob, self.ac) # , buffer_vae_mean, buffer_vae_std
        raw_act_p = self.buffer_model.decoder(self.next_ob, size = self.p, reuse = True) 
        raw_act_m = self.model.actor(self.ob, size = self.m, network='online_act')
        raw_act_n = self.buffer_model.decoder(self.ob, size = self.n, reuse = True)
        
        # initialize critic network
        self.q_func = self.model.critic(self.ob, self.ac, network='online_crt')
        """ raw_act_p """
        self.target_q_func = self.model.critic(tf.tile(self.next_ob, [self.p, 1]), raw_act_p, network='target_crt')
        
        # collect and define variables 
        self.q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'online_crt')
        self.target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'target_crt')
        self.policy_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'online_act')
        self.target_policy_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'target_act')
        self.buffer_policy_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'encoder')
        self.update_crt_vars = self.update_vars(self.q_func_vars, self.target_q_func_vars, tau = self.tau)
        self.update_act_vars = self.update_vars(self.policy_vars, self.target_policy_vars, tau = self.tau)
        with tf.variable_scope('dual'):
            self.dual_var = tf.get_variable('dual_var', initializer = tf.constant(np.random.randn(1,).astype('float32')),
                                            constraint=lambda x: tf.clip_by_value(x, -5, 10), dtype=tf.float32)
        self.dual_collection = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'dual')
        
        ### define VAE loss
        loss_recon = tf.reduce_mean(tf.reduce_sum(tf.square(buffer_vae_recon - self.ac), axis = 1))
        # kl_loss = - 0.5 * tf.reduce_sum((1 + tf.log(tf.square(buffer_vae_std)) - tf.square(buffer_vae_mean) - tf.square(buffer_vae_std)), axis = 1)
        self.vae_loss = loss_recon # + 0.5 * tf.reduce_mean(kl_loss) 
        # train VAE
        self.train_vae = tf.train.AdamOptimizer(self.learning_rate).minimize(self.vae_loss)
        
        ### compute target and train critic
        target = self.ld * tf.reduce_min(tf.reshape(self.target_q_func, [self.p, tf.shape(self.next_ob)[0], -1]), axis = 2)
        target += (1-self.ld)* tf.reduce_max(tf.reshape(self.target_q_func, [self.p, tf.shape(self.next_ob)[0], -1]), axis = 2)
        target = self.rew + self.gamma * (1-self.done) * tf.reduce_max(target, axis = 0)
        self.critic_loss = tf.reduce_sum(tf.reduce_mean(tf.square(self.q_func - tf.stop_gradient(tf.expand_dims(target, 1))), axis = 0))
        self.train_critic = tf.train.AdamOptimizer(self.learning_rate).minimize(self.critic_loss)
        
        ### train atcor using dual gradient descent
        actor_loss = self.model.critic(tf.tile(self.ob, [self.m, 1]), raw_act_m, network='online_crt', reuse = True)
        actor_loss = tf.reduce_mean(tf.reshape(actor_loss, [self.m, tf.shape(self.ob)[0], -1]), axis = 0) 
        std = self.var_ld * tf.sqrt(tf.clip_by_value(tf.math.reduce_std(actor_loss, axis = 1), 0, 10)) 
        actor_loss = tf.reduce_min(actor_loss, axis = 1) 
        self.mmd = tf.sqrt(mm_distance(raw_act_m, raw_act_n, (self.m, self.n), (self.batch_size, self.ac_dim), flags = self.flags) +1e-5)
        self.distance = tf.reduce_mean(self.mmd)
        self.actor_loss = tf.reduce_mean(-actor_loss + tf.stop_gradient(std) + tf.exp(self.dual_var)* (self.mmd-self.eps))
        act_gradients = tf.train.AdamOptimizer(self.learning_rate).compute_gradients(self.actor_loss, var_list = self.policy_vars)
        self.train_actor = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(act_gradients)
        self.dual_loss = tf.reduce_mean(-actor_loss + std + tf.exp(self.dual_var) * (self.mmd - self.eps))
        dual_grad = tf.train.AdamOptimizer(self.learning_rate).compute_gradients(-self.dual_loss, var_list = self.dual_collection)
        self.train_lagrange= tf.train.AdamOptimizer(self.learning_rate).apply_gradients(dual_grad)
        
    def update_vars(self, online_vars, target_vars, tau):
        """ update rate """
        update_target_fn = []
        for var, var_target in zip(sorted(online_vars, key=lambda v: v.name),
                                   sorted(target_vars, key=lambda v: v.name)):
            update_target_fn.append(var_target.assign(tau*var + (1-tau)*var_target))
        update_target_fn = tf.group(*update_target_fn)
        return update_target_fn
        
    def train(self, iterations):
        #####
        sess = get_session()
        set_seed(self.flags["seed"]) #, self.env)
        self.construct_graph()
        
        ##### log results using tensorboard
        with tf.name_scope('losses'):
            tf_vae_loss_summary = tf.summary.scalar('vae_loss', self.vloss)
            tf_crt_loss_summary = tf.summary.scalar('crt_loss', self.closs)
            tf_act_loss_summary = tf.summary.scalar('act_loss', self.aloss)
            tf_dual_loss_summary = tf.summary.scalar('dual_loss', self.dloss)
        
        tf_mmd_summary = tf.summary.scalar('mmd', self.distance)
        """ what is the usage of VAE? """
        summaries = tf.summary.merge([tf_vae_loss_summary, tf_crt_loss_summary, tf_act_loss_summary,
                                      tf_dual_loss_summary, tf_mmd_summary])
        self.tf_writer = tf.summary.FileWriter(self.flags["logdir"], sess.graph)
        stats = progress_stats(self.flags["logdir"])
        
        ##### initialize variables and assign online vars to target vars
        sess.run(tf.global_variables_initializer())    
        sess.run([self.update_vars(self.q_func_vars, self.target_q_func_vars, tau = 1), 
                  self.update_vars(self.policy_vars, self.target_policy_vars, tau = 1)])
        
        ##### train BEAR
        loss_his, reward_his = [], []
        self.t, t1 = 0, time.time()
        for it in range(iterations):
            # Load experience from buffer
            s, ns, a, r = self.buffer.sample(self.batch_size) # , done 
            
            # train vae and critic
            vae_loss, _ = sess.run([self.vae_loss, self.train_vae], feed_dict = {self.ob:s, self.ac:a})
            critic_loss, _ = sess.run([self.critic_loss, self.train_critic],
                                     feed_dict = {self.ob:s, self.ac:a, self.rew:r, self.done:done, self.next_ob:ns})
            
            #train actor using dual gradient descent
            act_loss, _ = sess.run([self.actor_loss, self.train_actor], feed_dict = {self.ob:s})
            dual_loss, mmd,  _ = sess.run([self.dual_loss, self.distance, self.train_lagrange], feed_dict = {self.ob:s})
            
            # update target vars
            sess.run([self.update_crt_vars, self.update_act_vars])
            
            # run evaluation and log results
#             if it % self.eval_freq == 0:
#                 t2 = time.time()
#                 self.run_evaluation(sess=sess)
#                 stats.save_csv()
#                 print('It: ' + str(it/1000) + ' Critic loss: ' + str(round(critic_loss, 3)) + ' Actor Loss: ' + str(round(act_loss, 3)) + ' Dual loss: ' + str(round(dual_loss, 3)) + ' Runtime (minute): ' + str(round((t2-t1)/60,3))) # ' MMD: ' + str(round(md,3))
#                 print('---------------------------------------------------------')
             
            stats.add(it, vae_loss, critic_loss, act_loss, dual_loss, mmd, self.reward_his[-1][0] if it % self.eval_freq == 0 else None,
                      self.reward_his[-1][1] if it % self.eval_freq == 0 else None)
            summ = sess.run(summaries, feed_dict = {self.vloss:vae_loss, self.closs:critic_loss, self.aloss:act_loss,
                                                    self.dloss: dual_loss, self.distance:mmd})
            self.tf_writer.add_summary(summ, self.t)
            self.t += 1
            
    def build_action_selection(self):
        """ ? test part ? 
        need to be modified for our purpose. Direct to the actor. 
        """
        action_samples, _ = self.model.actor(self.ob, size = self.m, network='online_act', reuse = True) 
        action_idx = tf.reduce_sum(self.model.critic(tf.tile(self.ob, [self.m, 1]), action_samples, network='online_crt',reuse = True)* tf.one_hot(tf.zeros(shape = tf.shape(action_samples)[0], dtype = tf.int32), 2, dtype='float32'), axis = 1)  # k = 2
        action_idx = tf.argmax(tf.reshape(action_idx, [self.m, -1]), axis = 0) 
        self.action = tf.reduce_sum(tf.transpose(tf.reshape(action_samples, [self.m, tf.shape(self.ob)[0], -1]), perm = [1,0,2]) * tf.expand_dims(tf.one_hot(action_idx, self.m, dtype='float32'), -1), axis = 1)
        
#     def run_evaluation(self, sess, num_eval=10):
#         if self.t == 0:
#             self.build_action_selection()
#             self.reward_his = []
#             self.eval_reward_log = tf.placeholder(dtype=tf.float32)
#             self.tf_eval_reward_log = tf.summary.scalar('eval_reward', self.eval_reward_log)
#             print('---------------------------------------------------------')
#             print('Train BEAR on ' + self.flags.game+'-'+ self.flags.version + ' with ' + self.flags.buffer)
#             print('---------------------------------------------------------')
            
#         reward = []
#         for _ in range(num_eval):
#             s = self.env.reset()
#             self.env.seed(self.t+100)
#             done = False
#             length_count = 0
#             total_reward = 0
#             while done == False:
#                 a = sess.run(self.action, feed_dict={self.ob:s.reshape(1, -1)})
#                 ns, r, done, info = self.env.step(a)
#                 total_reward += r
#                 s = ns
#             reward.append(total_reward)
#         self.reward_his.append((np.mean(reward), np.std(reward)))
#         avg_rew = sess.run(self.tf_eval_reward_log, feed_dict={self.eval_reward_log:np.mean(reward)})
#         self.tf_writer.add_summary(avg_rew, self.t)
#         print('Evaluation Reward mean: {0}, Reward std: {1}'.format(round(self.reward_his[-1][0], 3), round(self.reward_his[-1][1], 3)))

