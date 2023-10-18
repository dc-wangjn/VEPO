import numpy as np
import tensorflow.compat.v1 as tf
from time import sleep
import sys
import time

from networks import F_Network,W_Network
training_batch_size = 32
training_maximum_iteration = 3001
TEST_NUM = 20
class Density_Ratio_kernel_discounted(object):
    def __init__(self, obs_dim, w_hidden, Learning_rate, gamma):
        self.gamma = gamma
#         self.env = env
        
        # placeholder
        self.M = tf.placeholder(tf.float32,[])
        self.s0 = tf.placeholder(tf.float32, [None, obs_dim])
        
        self.state = tf.placeholder(tf.float32, [None, obs_dim])
        self.med_dist = tf.placeholder(tf.float32, [])
        self.next_state = tf.placeholder(tf.float32, [None, obs_dim])

        self.state2 = tf.placeholder(tf.float32, [None, obs_dim])
        self.next_state2 = tf.placeholder(tf.float32, [None, obs_dim])
        self.policy_ratio = tf.placeholder(tf.float32, [None])
        self.policy_ratio2 = tf.placeholder(tf.float32, [None])

        # density ratio for state and next state
#         with tf.variable_scope('w', reuse = tf.AUTO_REUSE):
        self.w_network = W_Network(w_hidden,'w')
        w = self.w_network(self.state)
        w2 = self.w_network(self.state2)
        w_next = self.w_network(self.next_state)
        w_next2 = self.w_network(self.next_state2)
        w_s0 = self.w_network(self.s0)
        norm_w = tf.reduce_mean(w)
        norm_w_next = tf.reduce_mean(w_next)
        norm_w_beta = tf.reduce_mean(w * self.policy_ratio)
        norm_w2 = tf.reduce_mean(w2)
        norm_w_next2 = tf.reduce_mean(w_next2)
        norm_w_beta2 = tf.reduce_mean(w2 * self.policy_ratio2)
        self.output = w


        x = self.gamma * w * self.policy_ratio / norm_w - w_next / norm_w_next
        self.dexf = self.gamma * w * self.policy_ratio / norm_w
        self.dexs = w_next / norm_w_next
        self.w = w 
        self.wn = w_next
        self.normwn =  norm_w_next
        
#         print('w shape {},norm_w shape{},pr {},w_next shape{}'.format(w.shape,norm_w.shape,self.policy_ratio.shape,w_next.shape))
        x2 = self.gamma * w2 * self.policy_ratio2 / norm_w2 - w_next2 / norm_w_next2
        self.dex2f = self.gamma * w2 * self.policy_ratio2 / norm_w2
        self.dex2s = w_next2 / norm_w_next2
        self.w2 = w2 
        self.w2n = w_next2
        diff_xx = tf.expand_dims(self.next_state, 1) - tf.expand_dims(self.next_state2, 0)
        self.K_xx = tf.exp(-tf.reduce_sum(tf.square(diff_xx), axis = -1)/(2.0*self.med_dist*self.med_dist))
        norm_K = tf.reduce_mean(self.K_xx)+0.001
#         print('x shape {},K_xx shape{},x2 shape{}'.format(x.shape,K_xx.shape,x2.shape))
        self.first_term = tf.matmul(tf.matmul(tf.expand_dims(x, 0),self.K_xx),tf.expand_dims(x2, 1))/(self.M*self.M)
        
        diff_xx2 = tf.expand_dims(self.next_state, 1) - tf.expand_dims(self.s0, 0)
        self.K_xx2 = tf.exp(-tf.reduce_sum(tf.square(diff_xx2), axis = -1)/(2.0*self.med_dist*self.med_dist))
        norm_K2 = tf.reduce_mean(self.K_xx2)+0.001
        inte_K = tf.reduce_mean(self.K_xx2,axis=1)
        self.second_term = tf.matmul(tf.expand_dims(x,0),tf.expand_dims(inte_K,1))*2*(1-self.gamma)/self.M
        
        diff_xx3 = tf.expand_dims(self.s0, 1) - tf.expand_dims(self.s0, 0)
        self.K_xx3 = tf.exp(-tf.reduce_sum(tf.square(diff_xx3), axis = -1)/(2.0*self.med_dist*self.med_dist))
        norm_K3 = tf.reduce_mean(self.K_xx3)+0.001
        self.third_term = tf.square(1-self.gamma)*tf.reduce_mean(self.K_xx3)
        self.normk3 = norm_K3
        
        self.f = tf.squeeze(self.first_term)/norm_K
        self.s = tf.squeeze(self.second_term)/norm_K2
        self.t = tf.squeeze(self.third_term)/norm_K3
        # self.loss = tf.squeeze(loss_xx)/(norm_w*norm_w2*norm_K)
#         self.loss = tf.squeeze(self.first_term)/norm_K + tf.squeeze(self.second_term)/norm_K2 + tf.squeeze(self.third_term)/norm_K3
        self.loss = tf.square(tf.squeeze(self.first_term)/norm_K + tf.squeeze(self.second_term)/norm_K2 + tf.squeeze(self.third_term)/norm_K3)
        self.reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'w'))
        self.train_op = tf.train.AdamOptimizer(Learning_rate).minimize(self.loss )

        # Debug
        self.debug1 = tf.reduce_mean(w)
        self.debug2 = tf.reduce_mean(w_next)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def reset(self):
        self.sess.run(tf.global_variables_initializer())

    def close_Session(self):
        tf.reset_default_graph()
        self.sess.close()


    def get_density_ratio(self, states):
        return self.sess.run(self.output, feed_dict = {
            self.state : states
            })

    def train(self, parms,SASR, p_r, batch_size = training_batch_size, max_iteration = training_maximum_iteration, test_num = TEST_NUM, fPlot = False, epsilon = 1e-3):
        S = []
        SN = []
        # POLICY_RATIO = []
        # POLICY_RATIO2 = []
        REW = []
        S0 = []
        
        for sasr in SASR:
            S0.append(sasr[0][0])
            for state, action, next_state, reward in sasr:
                # POLICY_RATIO.append((epsilon + policy1.pi(state, action))/(epsilon + policy0.pi(state, action)))
                # POLICY_RATIO2.append(policy1.pi(state, action)/policy0.pi(state, action))
                #POLICY_RATIO.append(epsilon + (1-epsilon) * policy1.pi(state, action)/policy0.pi(state, action))
                S.append(state)
                SN.append(next_state)
                REW.append(reward)
        # normalized
        S = np.array(S,dtype=np.float)  
        SN = np.array(SN,dtype=np.float)
        S0 = np.array(S0, dtype = np.float)
        S_max = np.max(S, axis = 0)
        S_min = np.min(S, axis = 0)
        S = (S - S_min)/(S_max - S_min)
        SN = (np.array(SN) - S_min)/(S_max - S_min)
        S0 = (S0 - S_min)/(S_max - S_min)


        if test_num > 0:
            S_test = np.array(S[:test_num])
            SN_test = np.array(SN[:test_num])
            # POLICY_RATIO_test = np.array(POLICY_RATIO[:test_num])
#             PI1_test = np.array(PI1[:test_num])
#             PI0_test = np.array(PI0[:test_num])
            p_r_test = np.array(p_r[:test_num])
        S = np.array(S[test_num:])
        SN = np.array(SN[test_num:])
        # POLICY_RATIO = np.array(POLICY_RATIO[test_num:])
        # POLICY_RATIO2 = np.array(POLICY_RATIO2[test_num:])
        p_r = np.array(p_r[test_num:])
        REW = np.array(REW[test_num:])
        N = len(S)

        subsamples = np.random.choice(N, 1000)
        s = S[subsamples]
# 		print(s)
        med_dist = np.median(np.sqrt(np.sum(np.square(s[None, :, :] - s[:, None, :]), axis = -1)))
        tloss = []

        for i in range(max_iteration):
            
            idx = np.random.permutation(N)
            S1 = S[idx]
            SN1 = SN[idx]
            p_r1 = p_r[idx]
            REW1 = REW[idx]

            S2 = S1[::-1]
            SN2 = SN1[::-1]
            p_r2 = p_r1[::-1]
            REW2 = REW1[::-1]
            for k in range(int(N/batch_size)):
                
    #             subsamples = np.random.choice(N, batch_size)
                s = S1[k*batch_size:(k+1)*batch_size]
                sn = SN1[k*batch_size:(k+1)*batch_size]
                # policy_ratio = POLICY_RATIO[subsamples]
                policy_ratio = p_r1[k*batch_size:(k+1)*batch_size]

#                 subsamples = np.random.choice(N, batch_size)
                s2 = S2[k*batch_size:(k+1)*batch_size]
                sn2 = SN2[k*batch_size:(k+1)*batch_size]
                # policy_ratio2 = POLICY_RATIO[subsamples]
                policy_ratio2 = p_r2[k*batch_size:(k+1)*batch_size]
    # 			print(med_dist,s,sn,policy_ratio,s2,sn2,policy_ratio2)
                self.sess.run([self.train_op], feed_dict = {
                    self.med_dist: med_dist,
                    self.state: s,
                    self.next_state: sn,
                    self.policy_ratio: policy_ratio,
                    self.state2 : s2,
                    self.next_state2: sn2,
                    self.policy_ratio2: policy_ratio2,
                    self.M: batch_size,
                    self.s0: S0
                    })
                loss, f, s, t, reg_loss, first, second, third,K_xx,K_xx2,K_xx3,normk3,dexf,dexs,dex2f,dex2s,w,w2,wn,w2n,normwn= self.sess.run([self.loss,self.f,self.s,self.t,self.reg_loss,self.first_term,self.second_term,self.third_term,self.K_xx,self.K_xx2,self.K_xx3,self.normk3,self.dexf,self.dexs,self.dex2f,self.dex2s,self.w,self.w2,self.wn,self.w2n,self.normwn], feed_dict = {
                    self.med_dist: med_dist,
                    self.state: s,
                    self.next_state: sn,
                    self.policy_ratio: policy_ratio,
                    self.state2 : s2,
                    self.next_state2: sn2,
                    self.policy_ratio2: policy_ratio2,
                    self.M: batch_size,
                    self.s0: S0
                    })
                if k%100==0:
                    print('epoch:{}, batch:{} ,loss:{}'.format(i,k,loss))
    
#             idx = np.random.permutation(len(S_test))
            s_test = S_test
            sn_test = SN_test
            # policy_ratio_test = POLICY_RATIO_test[subsamples]
            policy_ratio_test = p_r_test

#                 subsamples = np.random.choice(test_num, batch_size)
            s_test2 = s_test[::-1]
            sn_test2 = sn_test[::-1]
            # policy_ratio_test2 = POLICY_RATIO_test[subsamples]
            policy_ratio_test2 = policy_ratio_test[::-1]
# 				print(med_dist,s_test,sn_test,policy_ratio_test,s_test2,sn_test2,policy_ratio_test2)
            test_loss, reg_loss, norm_w, norm_w_next = self.sess.run([self.loss, self.reg_loss, self.debug1, self.debug2], feed_dict = {
                self.med_dist: med_dist,
                self.state: s_test,
                self.next_state: sn_test,
                self.policy_ratio: policy_ratio_test,
                self.state2: s_test2,
                self.next_state2: sn_test2,
                self.policy_ratio2: policy_ratio_test2,
                self.M: batch_size,
                self.s0: S0
                })
            tloss.append(test_loss)
            print('----Iteration = {}-----'.format(i))
            print("Testing error = {}".format(test_loss))
#             print('Regularization loss = {}'.format(reg_loss))
#             print('Norm_w = {}'.format(norm_w))
#             print('Norm_w_next = {}'.format(norm_w_next))
            DENR = self.get_density_ratio(S)
            # T = DENR*POLICY_RATIO2
#                 T = DENR*PI1/PI0
#                 print('DENR = {}'.format(np.sum(T*REW)/np.sum(T)))
            print('DENR = {}'.format(DENR))
            sys.stdout.flush()
                # epsilon *= 0.9
        nt,ts,gm,J,wh = parms[0],parms[1],parms[2],parms[3],parms[4]
        self.saver.save(self.sess,                                 'model/kernel_triple_nt{}_ts{}_gm{}_J{}_wh{}'.format(nt,ts,gm,J,wh))

#             if i<10:
#                 print('iteration:{},loss:{},f:{},s:{},t:{},reg:{},first:{},second:{},third:{},normk3:{},dexf:{},dexs:{},dex2f:{},dex2s:{},w:{},w2:{},wn:{},w2n:{},normwn:{},pr:{},wnmax:{},wnmin:{}'.format(i,loss,f,s,t,reg_loss,first,second,third,normk3,dexf[:10],dexs[:10],dex2f[:10],dex2s[:10],w[:10],w2[:10],wn[:10],w2n[:10],normwn,policy_ratio[:10],np.max(wn),np.min(wn)))

#         DENR = self.get_density_ratio(S)
#         # T = DENR*POLICY_RATIO2
#         T = DENR*PI1/PI0

        
        return tloss,DENR,S_max,S_min
#     def load(self, filename):
        
        
    def evaluate(self, SASR0, p_r,S_max,S_min):

        total_reward = 0.0
        self_normalizer = 0.0

        k = 0

#         pi = np.roll(np.array(policy0),axis=0,shift=-1)
        for sasr in SASR0:
#             discounted_t = 1.0
            
            state = np.array([a[0] for a in sasr])
            action = np.array([a[1] for a in sasr])
            next_state = np.array([a[2] for a in sasr])
            reward = np.array([a[3] for a in sasr]).squeeze(1)
            ts = state.shape[0]
            
            
            policy_ratio = np.array(p_r)[k*ts:(k+1)*ts]
            
            discounted_t = np.logspace(0,ts-1,num=ts,base=self.gamma)
#             print('policy_ratio is {}'.format(policy_ratio[:10]))
            s = np.array(state)
            s = (s - S_min)/(S_max-S_min)
            density_ratio = self.get_density_ratio(s)
#             print('raw:{}'.format(density_ratio))
#             print('reshape:{}'.format(density_ratio.reshape(-1)))
#             print('mean:{}'.format(np.mean(density_ratio.reshape(-1))))
            density_ratio = density_ratio.reshape(-1)/np.mean(density_ratio.reshape(-1))

#             print('norm:{}'.format(density_ratio))

            
            total_reward += np.sum(density_ratio.reshape(-1)*policy_ratio*discounted_t*reward)
            self_normalizer += np.sum(density_ratio.reshape(-1)*policy_ratio*discounted_t)

            k+=1            


        return total_reward / self_normalizer

    def evaluate_double(self, SASR0, p_r,policy0,S_max,S_min,Q_func):
        

        i= 0 
        k = 0
        psi = 0.0
        self_normalizer = 0.0
        s0_idx = []
#         pi = np.roll(np.array(policy0),axis=0,shift=-1)
        for sasr in SASR0:
#             discounted_t = 1.0
            
            s0_idx.append(i)
            i+=1
            
            state = np.array([a[0] for a in sasr])
            action = np.array([a[1] for a in sasr])
            next_state = np.array([a[2] for a in sasr])
            reward = np.array([a[3] for a in sasr]).squeeze(1)
            ts = state.shape[0]
            
            policy_ratio = np.array(p_r)[k*ts:(k+1)*ts]
            
#             print('policy_ratio is {}'.format(policy_ratio[:10]))
            s = np.array(state)
            s = (s - S_min)/(S_max-S_min)
            density_ratio = self.get_density_ratio(s)
            density_ratio = density_ratio.reshape(-1)/np.mean(density_ratio.reshape(-1))
#             print('density_ratio is {}'.format(density_ratio[:20]))
            
            
#             a_s = np.array([np.random.choice(p.shape[0],1,p=p) for p in pi[k*ts:(k+1)*ts] ]).reshape(-1,1)
            a_s = np.array([np.random.choice(len(policy0),1,p=policy0) for i in range(ts) ]).reshape(-1,1)
    
            sa = np.hstack([state,action])
            nsa = np.hstack([next_state,a_s])
            if Q_func == False:
                psi+= np.sum(density_ratio*policy_ratio*reward)
                
            else:
                psi+= np.sum(density_ratio*policy_ratio*(reward+self.gamma*Q_func.predict(nsa)-Q_func.predict(sa)))
            self_normalizer += np.sum(density_ratio*policy_ratio)
            k+=1
#             for ss in sasr:
#     #                 POLICY_RATIO.append(policy1[state, action]/policy0[state, action])
#     #                 S.append(state)
#     #                 REW.append(reward)

                
#                 state, action, next_state, reward = ss[0],ss[1],ss[2],ss[3]
#                 policy_ratio = p_r[i]

#                 s = np.array(state)
#                 s = (s - S_min)/(S_max-S_min)
#                 density_ratio = self.get_density_ratio(s.reshape(1,-1))
#                 print(i)
#                 p = policy0[i+1]
#                 a_s = np.random.choice(p.shape[0],1,p=p)
                
#                 sa = np.concatenate([state,action]).reshape(1,-1)
#                 nsa = np.concatenate([next_state,a_s]).reshape(1,-1)
# #                 print('ns:{},as:{},s:{},a:{}'.format(next_state.shape,a_s.shape,state,action))
#                 psi += density_ratio*policy_ratio*(reward+self.gamma*Q_func.predict(nsa)-Q_func.predict(sa)) 
#                 self_normalizer += density_ratio*policy_ratio
                
#                 i+=1
#                 if i<10:
#                     print('kernel evaluate: s:{},density_ratio:{},policy_ratio:{},psi:{}'.format(s,density_ratio,policy_ratio,psi))
#                 total_reward += density_ratio.reshape(-1)*policy_ratio*discounted_t*reward
#                 self_normalizer += density_ratio.reshape(-1)*policy_ratio*discounted_t
#                 discounted_t *=self.gamma
        if Q_func == False:
            q_mean =0
        else:
            S0 = []
    #         pi0_s0 = np.array(policy0)[s0_idx]
            for sasr in SASR0:
                S0.append(sasr[0][0])
    #         Q_init = 0
            s0 = np.array(S0)
    #         a_s = np.array([np.random.choice(p.shape[0],1,p=p) for p in pi0_s0 ])
            a_s = np.array([np.random.choice(len(policy0),1,p=policy0) for i in range(len(S0)) ])
            sa = np.hstack([s0,a_s])
            q_mean = np.mean(Q_func.predict(sa))
#         for ss in S0:
#             p = pi0_s0[j]
#             a_s = np.random.choice(p.shape[0],1,p=p)
#             sa = np.concatenate([ss,a_s]).reshape(1,-1)
#             Q_init += Q_func.predict(sa)
#             j+=1
      
        print('psi:{},self_normalizer:{},qmean:{}'.format(psi,self_normalizer,q_mean))
        rew = psi/self_normalizer/(1-self.gamma)+q_mean   
        return rew
    
class Density_Ratio_GAN_discounted(object):
    def __init__(self, obs_dim, w_hidden, f_hidden, Learning_rate, gamma):
        self.gamma = gamma
#         self.env = env
        # place holder
        self.state = tf.placeholder(tf.float32, [None, obs_dim])
        self.next_state = tf.placeholder(tf.float32, [None, obs_dim])
        self.s0 = tf.placeholder(tf.float32,[None,obs_dim])
        #self.start_state = tf.placeholder(tf.float32, [None, obs_dim])
        self.policy_ratio = tf.placeholder(tf.float32, [None])
        #self.isStart = tf.placeholder(tf.float32, [None])
#         with tf.variable_scope('w', reuse = tf.AUTO_REUSE):
        self.w_network = W_Network(w_hidden,'w')
        w = self.w_network(self.state)
        w_next = self.w_network(self.next_state)
        w_s0 = self.w_network(self.s0)
        norm_w = tf.reduce_mean(w)+ 1e-15
        norm_w_next = tf.reduce_mean(w_next)+ 1e-15
        norm_w_s0 = tf.reduce_mean(w_s0)+ 1e-15
        self.output = w
        self.w_norm = w/norm_w
        
        # density ratio for state and next state
#         if gau == 0:
#             w = self.state_to_w_batch_norm(self.state, obs_dim, w_hidden)
#             w_next = self.state_to_w_batch_norm(self.next_state, obs_dim, w_hidden)
#             w_s0 = self.state_to_w_batch_norm(self.s0, obs_dim, w_hidden)
#         else:
#             w = self.state_to_w_gau_mix(self.state, obs_dim, w_hidden)
#             w_next = self.state_to_w_gau_mix(self.next_state, obs_dim, w_hidden)
#             w_s0 = self.state_to_w_gau_mix(self.s0, obs_dim, w_hidden)
#         with tf.variable_scope('f', reuse = tf.AUTO_REUSE): 
        self.f_network = F_Network(f_hidden,'f')
        f = self.f_network(self.state)
        f_next = self.f_network(self.next_state)
        f_s0 = self.f_network(self.s0)
        norm_f_next = tf.reduce_mean(f_next) + 1e-15
        norm_f_s0 = tf.reduce_mean(f_s0)+ 1e-15
        self.f_n = f_next
        self.f_n_norm = f_next/norm_f_next
        # calculate loss function
        # discounted case
        # x = (1-self.isStart) * w * self.policy_ratio + self.isStart * norm_w - w_next
#         x = w * self.policy_ratio/norm_w - w_next/norm_w_next
        x = w * self.policy_ratio/norm_w - w_next/norm_w_next
        
        self.first = self.gamma*tf.reduce_mean(x * f_next/norm_f_next)
        self.second = (1-self.gamma)*tf.reduce_mean((1-w_s0/norm_w_s0)*f_s0/norm_f_s0)
        self.loss = tf.square((self.gamma*tf.reduce_mean(x * f_next/norm_f_next)+(1-self.gamma)*tf.reduce_mean((1-w_s0/norm_w_s0)*f_s0/norm_f_s0)))
#         with tf.variable_scope('optimizer'): 
      
        optimizer = tf.train.AdamOptimizer(Learning_rate)
        f_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'f')
        w_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'w')
        #             print('f_vars list:{}'.format(f_vars))
        #             print('w_vars list:{}'.format(w_vars))
        self.train_op_f = optimizer.minimize(-self.loss, var_list = f_vars)
        self.train_op_w = optimizer.minimize( self.loss , var_list = w_vars)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def reset(self):
        self.sess.run(tf.global_variables_initializer())

    def close_Session(self):
        tf.reset_default_graph()
        self.sess.close()

    def get_density_ratio(self, states):
        return self.sess.run(self.output, feed_dict = {
            self.state : states
            })

    def train(self, SASR, p_r, batch_size = training_batch_size, max_iteration = training_maximum_iteration, test_num = TEST_NUM, fPlot = False):
        S = []
        SN = []
#         POLICY_RATIO = []
        REW = []
        S0 = []
        for sasr in SASR:
            S0.append(sasr[0][0])
            for state, action, next_state, reward in sasr:
                S.append(state)
                SN.append(next_state)
                REW.append(reward)
        S = np.array(S,dtype = np.float)
        SN = np.array(SN,dtype = np.float)
        S0 = np.array(S0, dtype = np.float) 
        print(S.shape)
# 		POLICY_RATIO = np.array(POLICY_RATIO).reshape(-1,1)         
        if test_num > 0:
            S_test = np.array(S[:test_num])
            SN_test = np.array(SN[:test_num])
            p_r_test = np.array(p_r[:test_num])
            
        S = np.array(S[test_num:])
        SN = np.array(SN[test_num:])
        p_r = np.array(p_r[test_num:])
        REW = np.array(REW[test_num:])

        N = len(S)
        for i in range(max_iteration):
            idx = np.random.permutation(N)
            S1 = S[idx]
            SN1 = SN[idx]
            p_r1 = p_r[idx]
            REW1 = REW[idx]
            for k in range(int(N/batch_size)):
                s = S1[k*batch_size:(k+1)*batch_size]
                sn = SN1[k*batch_size:(k+1)*batch_size]
                # policy_ratio = POLICY_RATIO[subsamples]
                policy_ratio = p_r1[k*batch_size:(k+1)*batch_size]
                for t in range(10):
                    self.sess.run(self.train_op_f, feed_dict = {
                        self.state: s,
                        self.next_state: sn,
                        self.policy_ratio: policy_ratio,
                        self.s0:S0
                        })

                self.sess.run(self.train_op_w, feed_dict = {
                    self.state: s,
                    self.next_state: sn,
                    self.policy_ratio: policy_ratio,
                    self.s0:S0
                    }) 
                loss,w,w_norm,fn, fn_norm,first,second = self.sess.run([self.loss,self.output,self.w_norm,self.f_n,self.f_n_norm,self.first,self.second], feed_dict = {
                    self.state: s,
                    self.next_state: sn,
                    self.policy_ratio: policy_ratio,
                    self.s0:S0
                    }) 
#                 if k%10==0:
#                     print('epoch:{}, batch:{} ,loss:{}, w:{},w_norm:{},fn:{}, fn_norm:{},first:{},second:{}'.format(i,k,loss,w[:10],w_norm[:10],fn[:10], fn_norm[:10],first,second))
            idx = np.random.permutation(len(S_test))
            s_test = S_test[idx][:batch_size]
            sn_test = SN_test[idx][:batch_size]
            # policy_ratio_test = POLICY_RATIO_test[subsamples]
            policy_ratio_test = p_r_test[idx][:batch_size]
            
            test_loss = self.sess.run([self.loss], feed_dict = {
                self.state: s_test,
                self.next_state: sn_test,
                self.policy_ratio: policy_ratio_test,
                self.s0: S0
                })
            print('----Iteration = {}-----'.format(i))
            print("Testing error = {}".format(test_loss))
#             print('Regularization loss = {}'.format(reg_loss))
#             print('Norm_w = {}'.format(norm_w))
#             print('Norm_w_next = {}'.format(norm_w_next))
            DENR = self.get_density_ratio(S)
            print('DENR = {}'.format(DENR))
            sys.stdout.flush()



    def evaluate(self, SASR, p_r):
        S = []
        REW = []
#         policy_ratio = []
        total_reward = 0.0
        self_normalizer = 0.0
#         w_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'w')
#         for x in w_vars:
#             print('variable :{}'.format(self.sess.run(x)))
#         print(w_vars)
        i= 0 
        for sasr in SASR:
            discounted_t = 1.0
            for state, action, next_state, reward in sasr:
                policy_ratio = p_r[i]
                s = np.array(state)
                
                density_ratio = self.get_density_ratio(s.reshape(1,-1))
                
#                 print('state:{},decoding:{},shape:{},density_ratio:{}'.format(state,s,s.shape,density_ratio))
                i += 1
                if i<10 :
                    print('gan evaluate: s:{},density_ratio:{},policy_ratio:{}'.format(s,density_ratio,policy_ratio))
                total_reward += density_ratio.reshape(-1)*policy_ratio*discounted_t*reward
                self_normalizer += density_ratio.reshape(-1)*policy_ratio*discounted_t
                discounted_t *=self.gamma

        return total_reward / self_normalizer
    def evaluate_double(self, SASR, p_r,policy0,Q_func):

        i= 0 
        k = 0
        psi = 0.0
        self_normalizer = 0.0
        s0_idx = []

        for sasr in SASR:
            s0_idx.append(i)
            i+=1

            state = np.array([a[0] for a in sasr])
            action = np.array([a[1] for a in sasr])
            next_state = np.array([a[2] for a in sasr])
            reward = np.array([a[3] for a in sasr]).squeeze(1)
            ts = state.shape[0]
            
            policy_ratio = np.array(p_r)[k*ts:(k+1)*ts]
            s = np.array(state)
            density_ratio = self.get_density_ratio(s)
            a_s = np.array([np.random.choice(len(policy0),1,p=policy0) for i in range(ts) ]).reshape(-1,1)
            sa = np.hstack([state,action])
            nsa = np.hstack([next_state,a_s])
            
            if Q_func == False:
                psi+= np.sum(density_ratio*policy_ratio*reward)
                
            else:
                psi+= np.sum(density_ratio*policy_ratio*(reward+self.gamma*Q_func.predict(nsa)-Q_func.predict(sa)))
            self_normalizer += np.sum(density_ratio*policy_ratio)
            k+=1            

                
        if Q_func == False:
            q_mean =0
        else:
            S0 = []
    #         pi0_s0 = np.array(policy0)[s0_idx]
            for sasr in SASR:
                S0.append(sasr[0][0])
    #         Q_init = 0
            s0 = np.array(S0)
    #         a_s = np.array([np.random.choice(p.shape[0],1,p=p) for p in pi0_s0 ])
            a_s = np.array([np.random.choice(len(policy0),1,p=policy0) for i in range(len(S0)) ])
            sa = np.hstack([s0,a_s])
            q_mean = np.mean(Q_func.predict(sa))
#         for ss in S0:
#             p = pi0_s0[j]
#             a_s = np.random.choice(p.shape[0],1,p=p)
#             sa = np.concatenate([ss,a_s]).reshape(1,-1)
#             Q_init += Q_func.predict(sa)
#             j+=1
      
        print('psi:{},self_normalizer:{},qmean:{}'.format(psi,self_normalizer,q_mean))
        rew = psi/self_normalizer/(1-self.gamma)+q_mean   
        return rew

    def learning_distribution(self, S0, S1, S11, policy0, policy1, dim_ID):
        w = self.get_density_ratio(np.array(S0))
        #print(w)
        #print(w[np.isnan(w)])
        # Only consider the marginal distribution of the first feature of state
        N = int(max(np.amax(S0[:,dim_ID]), np.amax(S1[:,dim_ID]), np.amax(S11[:, dim_ID]))) + 1

        p1_true = np.zeros(N, dtype = np.float32)
        p1_true_copy = np.zeros(N, dtype = np.float32)
        p1_estimate = np.zeros(N, dtype = np.float32)
        p0_true = np.zeros(N, dtype = np.float32)

        for s in S1:
            index = int(s[dim_ID])
            p1_true[index] += 1.0

        for s in S11:
            index = int(s[dim_ID])
            p1_true_copy[index] += 1.0

        for wi,s in zip(w,S0):
            index = int(s[dim_ID])
            p0_true[index] += 1.0
            p1_estimate[index] += wi
        return p1_true/np.sum(p1_true), p1_estimate/np.sum(p1_estimate), p0_true/np.sum(p0_true), p1_true_copy/np.sum(p1_true_copy)