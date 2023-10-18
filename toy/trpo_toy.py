import numpy as np
import time
from livelossplot import PlotLosses
import tensorflow as tf

class TOY():
    def __init__(self, gamma, delta,init_lamb,r_sigma):
        self.state_dim = 2
        self.action_dim = 2

        self.gamma = gamma
        self.KL_delta = delta
        self.r_sigma = r_sigma


        self.trans = np.array([[0.75, 0.25], [0.4, 0.6],[0.1,0.9],[0.85,0.15]])
        self.pm = np.array([[0.7,0.3],[0.2,0.8]])
        self.init_dist = np.array([0.6,0.4])
        
        self.init_lamb = init_lamb

    def init_sampler(self, num_trajs):
        init_state = np.random.choice([0,1],num_trajs,p=self.init_dist)
        return init_state

    def step(self, state, action, trans=None):
        if trans is None :
            trans = self.trans
        state = state.astype(np.int)
        action = action.astype(np.int)
        ind = (state<<1)+action
        prob = trans[ind]
        nn = state.shape[0]

        next_state = np.zeros(nn)
        r = np.random.rand(nn)

        next_state[r > prob[:, 0]] = 1
        next_state = next_state.astype(np.int).squeeze()
        # next_state = np.array([np.random.choice([0,1],1,p = p) for p in prob]).squeeze()
        return next_state

    # def behav_policy(self,state):
    #     prob = self.policy[state]
    #     action = np.array([np.random.choice([0,1],1,p = p) for p in prob]).squeeze()
    #     return  action
    def make_policy(self, pm ):
        def policy(state):
            prob = pm[state]
            nn = state.shape[0]
            action = np.zeros(nn)
            r = np.random.rand(nn)

            action[r>prob[:,0]] =1
            action = action.astype(np.int).squeeze()
            # action = np.array([np.random.choice([0, 1], 1, p=p) for p in prob]).squeeze()
            return action
        return policy

    def reward(self, state, action, noise_on = False):
#         num_trajs = state.shape[0]

        reward = np.zeros_like(state)
        reward[state!=action] = 1
        if noise_on == True:
            
            noise = np.random.normal(0,self.r_sigma,reward.shape)
            reward = reward + noise
        return reward

    def sample_init_state(self, num_trajs):
        return self.init_sampler(num_trajs)

    def sample_trajectory(self,num_trajs,tt, pm=None,trans=None):
        if pm is None:
            # print('use improved policy!')
            policy = self.new_pi.sample_A
        else:
            # print('use assigned matrix for policy!')
            policy = self.make_policy(pm)
        trajs = np.zeros((num_trajs, tt, 2))
        s = self.init_sampler(num_trajs)
#         print('sample_trajs using trans {}'.format(trans))
        for t in range(tt):
            a = policy(s)
            trajs[:,t] = np.concatenate([s.reshape(-1,1),a.reshape(-1,1)],axis=1)
            s = self.step(s,a,trans)
        return trajs

    def sample_cond_trajectory(self, M ,tt, pm, state, action, trans=None):
        if pm is None:
            # print('use improved policy!')
            policy = self.new_pi.sample_A
        else:
            # print('use assigned matrix for policy!')
            policy = self.make_policy(pm)
        tn = state.shape[0]
        trajs = np.zeros((tn * M, tt, 2))
        s = np.repeat(state,M)
        a = np.repeat(action,M)
        trajs[:, 0] = np.concatenate([s.reshape(-1,1),a.reshape(-1,1)],axis=1)
#         print('sample_trajs using trans {}'.format(trans))
        for t in range(1,tt):
            s = self.step(s, a, trans)
            a = policy(s)
            trajs[:,t] = np.concatenate([s.reshape(-1,1),a.reshape(-1,1)],axis=1)

        # trajs = trajs.reshape(tn, M, tt, 2)
        return trajs

    def cal_q_function(self, pm, num_trajs, M,tt):
        if pm is None:
            s = np.array([0, 1])
            pm = self.new_pi.get_A_prob(s)
        q = np.zeros((self.state_dim,self.action_dim))
        for s in range(self.state_dim):
            for a in range(self.action_dim):
#                 print('now proc with state {} action {}'.format(s,a))
                s0 = s*np.ones(num_trajs)
                a0 = a*np.ones(num_trajs)
                trajs = self.sample_cond_trajectory(M, tt, pm, s0, a0)
                q[s,a] = np.mean(self.eval_trajs(trajs))

        return q

    def cal_A_function(self,pm, num_trajs, M,tt):

        if pm is None:
            s = np.array([0, 1])
            pm = self.new_pi.get_A_prob(s)
        q = self.cal_q_function(pm, num_trajs, M,tt)
        v = np.sum(q * pm, axis = 1).reshape(-1,1)
        return q - v , v.reshape(-1)


    def cal_density_ratio(self, num_trajs, M,tt, pm_old, pm_b):
        if pm_old is None:
            s = np.array([0, 1])
            pm_old = self.new_pi.get_A_prob(s)

        w = np.zeros((self.state_dim , self.action_dim))
        trajs_pb = self.sample_trajectory(num_trajs, tt, pm_b)
        trajs_pold = self.sample_trajectory(num_trajs, tt, pm_old)
        for s in range(self.state_dim):
            for a in range(self.action_dim):
                s_old = (trajs_pold[:,:,0]==s)
                a_old = (trajs_pold[:,:,1]==a)
                d_old = np.mean(s_old * a_old)
                # print(d_old)
                s_b = (trajs_pb[:,:,0]==s)
                a_b = (trajs_pb[:,:,1]==a)
                d_b = np.mean(s_b * a_b)
                # print(d_b)
                w[s,a] = d_old/d_b

        w_cond = np.zeros((self.state_dim*self.action_dim , self.state_dim*self.action_dim))
        for b in range(self.state_dim*self.action_dim):
#             print('now proc with condition {}'.format(b))
            s_b = b >> 1
            a_b = b - (s_b << 1)
            state = (s_b * np.ones(num_trajs)).astype(np.int)
            action = (a_b * np.ones(num_trajs)).astype(np.int)
            trajs_pold = self.sample_cond_trajectory(M, tt, pm_old, state, action)
            for t in range(self.state_dim*self.action_dim):

                s_t = t>>1
                a_t = t - (s_t<<1)

                s_old = (trajs_pold[:,:,0]==s_t)
                a_old = (trajs_pold[:,:,1]==a_t)
                d_old = np.mean(s_old * a_old)
                # print(d_old)
                s_b = (trajs_pb[:,:,0]==s_t)
                a_b = (trajs_pb[:,:,1]==a_t)
                d_b = np.mean(s_b * a_b)
                # print(d_b)
                w_cond[t,b] = d_old/d_b
        return w, w_cond
    
    def get_A(self, q, pm_old):
        v = np.sum(q * pm_old, axis = 1).reshape(-1,1)
        return q - v , v.reshape(-1)
    
    def get_omega(self, omega_cond, pm_old, init_dist):
        init_dist = init_dist[:,None]
        pi = (pm_old * init_dist).reshape(1,-1,order='F')
        omega = np.sum(omega_cond*pi,axis=1)
        omega = omega.reshape(pm_old.shape)

        return omega

    def compute_components(self,  pm_old, cal_nu_nn,cal_nu_mm,cal_nu_tt,M,M2, tt, trajs_behav = None,
                     a_func= None, v_func = None ,omega = None, omega_cond = None,
                     trajs_star = None, trajs_cond_temp = None,trans=None):
        '''sholud use test set as trajs_behav instead of the whole trajs'''
        if a_func is None:
#             print('calculating A-function ...')
            a_func,v_func = self.cal_A_function(pm_old,cal_nu_nn,cal_nu_mm,cal_nu_tt)
        if omega is None:
#             print('calculating density ratio ... ')
            omega, omega_cond = self.cal_density_ratio(cal_nu_nn,cal_nu_mm,cal_nu_tt, pm_old ,self.pm)
        if pm_old is None:
            s = np.array([0, 1])
            pm_old = self.new_pi.get_A_prob(s)

        # n1*tt1
        state = trajs_behav[:,:-1,0].reshape(-1).astype(np.int)
        action = trajs_behav[:,:-1,1].reshape(-1).astype(np.int)
        next_state = trajs_behav[:,1:,0].reshape(-1).astype(np.int)
        reward = self.reward(state,action,True)

        self.nt = state.shape[0]
        nt = self.nt

        self.NN, self.TT = NN, TT
        self.TTNN = NN * TT
        TN = self.N * self.T
        

        N = len(self.split_ind[0]["test_ind"])
        gamma, L = self.gamma, self.L

        FACTOR = (M * (1 - gamma ** TT))
        gammas = arr([gamma ** t for t in range(TT)])
        broadcast_gammas = np.repeat(row_repeat(gammas, NN * TN, full_block=True), self.num_A, axis=1)
        broadcast_gammas_2 = np.repeat(np.expand_dims(np.tile(gammas, NN), 1), self.num_A, axis=1)

        gene_S_l1, gene_S_l1_one_old, gene_S_l2, gene_S_star_l2 = [], [], [], []
        old_probs_KL, term2_old_parts, term2_new_otherparts_with_gamma = [], [], []
        adv_l1_w_gamma = []
        adv_4_gene_S_star_l2, adv_4_gene_S_l2, adv_4_real_l2 = [], [], []
        l2_omega = []
        adv_next_S_l2, adv_S_l2 = [], []
        term2_news_omega_TD_gamma = []
        
        total_states = []

        ########################################################################
        ##################### calculate fixed data part ########################
        ########################################################################

        """ training data has been used, test_idx is then used in this part to construct the functions """
        curr_time = now()
        states, actions, rewards, next_states = [
            np.array([item[i] for traj in [self.trajs_ori[j] for j in split["test_ind"]] for item in traj]) for i in range(4)]
        TN_test = len(states)
        self.TN_test = TN_test
        self.TN = TN_test * self.L

        ################## GENERATE SAMPLES ##################
        if use_ori_trans:
            gene_S = self.ohio.sample_S_4VE(pi=self.curr_policy,NN=NN, TT=TT, seed = self.seed)
            gene_S = (gene_S - self.s_min)/(self.s_max - self.s_min)

            gene_cond_S, gene_cond_S_star = self.ohio.sample_cond_S_4VE(states, actions, next_states,self.curr_policy, M, TT ,seed = self.seed)
            gene_cond_S = (gene_cond_S - self.s_min)/(self.s_max - self.s_min)
            gene_cond_S_star = (gene_cond_S_star - self.s_min)/(self.s_max - self.s_min)

        else:
            gene_S = self.sample_S(self.init_sampler, self.curr_policy, G, NN, TT,self.S_dims)  # (TTNN, dim_S): shared within fold; can not shared across due to that G is different
            gene_cond_S, gene_cond_S_star = self.sample_cond_S(states, actions, next_states, M, TT, self.curr_policy, G)
        states, actions, rewards, next_states = [
            np.array([item[i] for traj in [self.trajs[j] for j in split["test_ind"]] for item in traj]) for i in range(4)]
        total_states.append(states)
        old_probs_KL.append(self.curr_policy.get_A_prob(gene_S, to_numpy=True))  # (TTNN, num_A)
        ####################################### term1: plug-in  #######################################
        adv_l1_w_gamma.append(old_value.A_func(gene_S) * broadcast_gammas_2)  # (TTNN, num_A)
        printG("<------  term 1 for fold {} DONE! Time cost = {:.1f} minutes -------->\n".format(k, (now() - curr_time) / 60))
        ################################# term2 (new) = l_1 (old) #######################################

        gene_S_l1.append(np.repeat(np.expand_dims(gene_S, 0), TN_test, 0))  # (TN, TTNN, dim_S)
        gene_S_l1_one_old.append(gene_S)  # (TTNN, dim_S)
        """ l1 term here; already gamma """
        term2_old_part, term2_new_omega_TD_gamma = self.l1_main_term(self.curr_policy, old_value, omega, gene_S,split)
        term2_old_parts.append(term2_old_part)  # scalar
        term2_news_omega_TD_gamma.append(term2_new_omega_TD_gamma)  # (TN, TTNN, num_A)
        printG("<------  term 2 for fold {} DONE! Time cost = {:.1f} minutes -------->\n".format(k, (now() - curr_time) / 60))
        ################################# term3 (new) = l_2 (old)#######################################
        #################### \tilde{omega}^*:  sample and average ####################
        # <------  sample for fold 0 DONE! Time cost = 0.5 minutes -------->
        # <------  3.1 for fold 0 DONE! Time cost = 4.5 minutes -------->
        # <------  3.2 for fold 0 DONE! Time cost = 8.6 minutes -------->
        l2_omega.append(self.l2_omega_term(G_ds_n_seeds, self.curr_policy, states, actions, omega))  # len = TN
        printG("<------  l2_omega_term for fold {} DONE! Time cost = {:.1f} minutes -------->\n".format(k, (
                    now() - curr_time) / 60))
        ################################################################################
        #### generate random samples; dim = (TN, M, TT, dim_S)

        gene_S_l2.append(gene_cond_S)
        gene_S_star_l2.append(gene_cond_S_star)
        ############################ 1; (TN, M, TT, num_A) ############################
        adv_next_S = old_value.A_func(gene_cond_S_star.reshape(-1, self.S_dims)).reshape(TN_test, M, TT,self.num_A) * gamma / FACTOR
        adv_next_S_l2.append(np.stack([adv_next_S[:, :, i, :] * gammas[i] for i in range(len(gammas))], axis=2))
        printG("<------  3.1 for fold {} DONE! Time cost = {:.1f} minutes -------->\n".format(k, (now() - curr_time) / 60))
        ############################ 2; (TN, M, TT, num_A) ############################
        adv_S = old_value.A_func(gene_cond_S.reshape(-1, self.S_dims)).reshape(TN_test, M, TT, self.num_A) / FACTOR
        adv_S_l2.append(np.stack([adv_S[:, :, i, :] * gammas[i] for i in range(len(gammas))], axis=2))
        printG("<------  3.2 for fold {} DONE! Time cost = {:.1f} minutes -------->\n".format(k, (now() - curr_time) / 60))
        ############################ 3 ############################
        repeat_gammas = np.tile([gamma ** t for t in range(0, TN_test // N)], N)
        adv_4_real_l2.append(
            np.dot(diag(repeat_gammas), old_value.A_func(states)))  # (TN, num_A); multiply with each row
        printG(
            "<------  lower bound construction for fold {} DONE! Time cost = {:.1f} minutes -------->\n".format(k, (now() - curr_time) / 60))
        #####################################################################################################

        self.components = {"adv_l1_w_gamma": arr(adv_l1_w_gamma)  # (L, TTNN, num_A)
                           # , "gene_S_l1" : np.vstack(gene_S_l1) # (TN, TTNN, dim_S)
            , "gene_S_l1_one_old": arr(gene_S_l1_one_old)  # (L, TTNN, dim_S)
            , "term2_news_omega_TD_gamma": arr(term2_news_omega_TD_gamma)  # (L, TTNN, num_A)
            , "term2_old_part": np.mean(term2_old_parts)
            , "gene_S_l2": np.vstack(gene_S_l2)  # (TN, M, TT, dim_S)
            , "total_states":np.vstack(total_states)
            , "gene_S_star_l2": np.vstack(gene_S_star_l2)  # (TN, M, TT, dim_S)
            , "adv_next_S_l2": np.vstack(adv_next_S_l2)  # (TN, M, TT, num_A)
            , "adv_S_l2": np.vstack(adv_S_l2)  # (TN, M, TT, num_A)
            , "adv_4_real_l2": np.vstack(adv_4_real_l2)  # (TN, num_A)
            , "l2_omega": np.concatenate(l2_omega)  # len = TN
            , "old_probs_KL": arr(old_probs_KL)  # (L, TTNN, num_A)
            , "broadcast_gamma_TTNN": np.tile(gammas, NN)  # (TTNN,)
                           }


    def compute_loss(self, real_idx = None):

 
        A_prob_gene_S = self.new_pi.get_A_prob(self.components["gene_S_l1_one_old"], multi_dim=True)
        FACTOR = tf.reduce_sum(self.components["broadcast_gamma_TTNN"]) * self.L
        l1 = tf.reduce_sum(self.components["adv_l1_w_gamma"] * A_prob_gene_S) / FACTOR

        ######################## l2 ########################
        """ no SGD; already average over NT """
        l2_old_part = self.components["term2_old_part"]
        # (L, TTNN, num_A)
        l2_new_part = tf.reduce_sum(self.components["term2_news_omega_TD_gamma"] * A_prob_gene_S) / FACTOR / (1 - self.gamma)

        l2 = l2_new_part - l2_old_part

        ########################################################################
        ######################## l3: most memory-intense #######################
        ########################################################################
        """ self.policy """
        """ the two most memory-consuming parts """
        # (TN, M, TT, dim_S)
        
        states = self.components["total_states"][real_idx]
        A_prob_gene_S = self.new_pi.get_A_prob(self.components["gene_S_l2"][real_idx], multi_dim=True)
        # (TN, M, TT, dim_S)
        A_prob_gene_S_star = self.new_pi.get_A_prob(self.components["gene_S_star_l2"][real_idx], multi_dim=True)
        # (TN, num_A)
        A_prob_S = self.new_pi.get_A_prob(states)
        # (TN, )
        l3 = tf.reduce_sum(self.gamma * A_prob_gene_S_star * self.components["adv_next_S_l2"][real_idx] - \
                           A_prob_gene_S * self.components["adv_S_l2"][real_idx], (1, 2, 3)) + \
             (1 - self.gamma) * tf.reduce_sum(A_prob_S * self.components["adv_4_real_l2"][real_idx], 1)
        l3 = l3 * self.components["l2_omega"][real_idx] / (1 - self.gamma)  # (TN, )
        l3 = tf.reduce_mean(l3)

        ######################## KL term; (L, TTNN, num_A) ########################
        # (L, TTNN, num_A)
        KL_divergence = self.kl(curr_probs=self.components["old_probs_KL"]
                                , target_probs=self.new_pi.get_A_prob(self.components["gene_S_l1_one_old"],
                                                                      multi_dim=True))
        KL_loss = tf.tensordot(self.components["broadcast_gamma_TTNN"], KL_divergence, 1) * (1 - self.gamma) / (
                    self.NN * self.L * (1 - self.gamma ** self.TT))
        ################################# LOSS #################################
        lift = l1 + l2 + l3

        loss = -lift + tf.nn.relu(KL_loss - self.KL_delta) * tf.math.exp(self.new_pi.loglam)
        
        
        
        return loss, lift, l1, l2, l3, KL_loss,tf.math.exp(self.new_pi.loglam)
    
    def kl(self, curr_probs, target_probs):
        return tf.reduce_sum(curr_probs * tf.math.log(
            tf.cast(tf.clip_by_value(curr_probs, 1e-6, 1), tf.float64) / tf.cast(
                tf.clip_by_value(target_probs, 1e-6, 1), tf.float64)), (0, 2))

    
    def construct_policy(self,simple, w,h_dims=2, lr_lambda=0.0005, Adam_para={"lr": 0.0005, "w_clipping_val": 0.5, "w_clipping_norm": 1.0}):
        

        self.new_pi = ToyPolicy(simple, init_weight = w,init_lamb = self.init_lamb, state_dim=1, action_dim=2, h_dims= h_dims)
        print('lambda is {}'.format(np.exp(self.new_pi.loglam.numpy().reshape(-1))))
        
        s = np.array([0, 1])
        up_pm_old = self.new_pi.get_A_prob(s)
        print('initial construct pi {}'.format(up_pm_old))
        
        self.optimizer = tf.keras.optimizers.Adam(Adam_para["lr"], beta_1=0.5, clipnorm=Adam_para["w_clipping_norm"],
                                                  clipvalue=Adam_para["w_clipping_val"])
        self.optimizer_lambda = tf.keras.optimizers.Adam(lr_lambda, beta_1=0.5, clipnorm=Adam_para["w_clipping_norm"],
                                                         clipvalue=Adam_para["w_clipping_val"])


    def improve_policy(self, simple, multi_step, max_iter=100, batch_size=32, print_freq=100,eps=1e-4):
        self.losses = []
        self.lift = []
        self.kl_losses = []
        self.lamb = []
        self.change_rates = []
        # groups = {'value': ['loss', 'lift', 'True Values', 'l3'], 'KL': ['KL_loss', 'loglam']}
        # plotlosses = PlotLosses(groups=groups)
        nbatch = int(self.nt/batch_size)
        old_value = -1000000
        if simple == True:
            parms = [self.new_pi.w1, self.new_pi.b1]
        else:
            parms = [self.new_pi.w1, self.new_pi.b1, self.new_pi.w2, self.new_pi.b2]
        for i in range(max_iter):
            if i%nbatch == 0:
                shuffled_index = np.random.permutation(self.nt)
            j = i%nbatch
            real_idx = shuffled_index[j*batch_size:(j+1)*batch_size]
#             real_idx = np.random.choice(self.nt, batch_size, replace=True)
            for j in range(multi_step):
                with tf.GradientTape() as tape1:

                    loss, lift ,l1 ,l2 , l3, kl_loss, lamb= self.compute_loss(real_idx)
                dw = tape1.gradient(loss, parms)
                # print(dw)
                self.optimizer.apply_gradients(zip(dw, parms))

#             with tf.GradientTape() as tape2:
#                 loss_lam, lift ,l1 ,l2 , l3, kl_loss, lamb= self.compute_loss(real_idx)
#                 loss_lam = -loss_lam
#             dw = tape2.gradient(loss_lam, [self.new_pi.loglam])
#             self.optimizer_lambda.apply_gradients(zip(dw, [self.new_pi.loglam]))
            self.lift.append(lift)
            self.kl_losses.append(kl_loss)
            self.lamb.append(lamb)
            self.losses.append(loss)

            change_rate = np.abs(np.mean(loss)-old_value)/np.abs(old_value + 1e-6)
#             change_rate = np.abs(np.mean(loss)-old_value)/np.abs(old_value + 1e-6)
            self.change_rates.append(change_rate)
#             print('change rate :{}'.format(change_rate))
            
            if i % print_freq == 0 and i >= print_freq:
                print('i :{} ,loss:{} ,lift:{}'.format(i,np.mean(self.losses[-1:]),np.mean(self.lift[-1:])))
                print('l1:{},l2:{},l3:{}'.format(l1.numpy(), l2.numpy(), l3.numpy()))
                #former idx (i-10):i
                print('kl loss :{}, lambda :{}'.format(np.mean(self.kl_losses[-1:]),np.mean(self.lamb[-1:])))
                s = np.array([0, 1])
                up_pm_old = self.new_pi.get_A_prob(s)
                print('last iter pi {}'.format(up_pm_old))
                print('change rate :{}'.format(change_rate))
                
#                 print('w1 is {}'.format(self.new_pi.w1))
#                 print('w2 is {}'.format(self.new_pi.w2))
#                 print('b1 is {}'.format(self.new_pi.b1))
#                 print('b2 is {}'.format(self.new_pi.b2))
                
                # discounted_values = self.eval_onpolicy(policy="improved", simulator="ohio", sd_G=sd_G, echo=False)
                # values.append(np.mean(discounted_values))
                # plotlosses.update({
                #     'loss': mean(self.losses[(i - 10):i]),
                #     'KL_loss': mean(self.KL_losses[(i - 10):i]),
                #     'loglam': mean(self.lams[(i - 10):i]),
                #     'True Values': np.mean(discounted_values),
                #     'lift': mean(self.lift[(i - 10):i]),
                #     'l3': mean(self.l3[(i - 10):i])
                # })
                # plotlosses.send()
            if np.mean(self.change_rates[-5:]) < eps :
                print('Reach the threshold and converge!  i:{} ;Change rate : {} '.format(i,np.mean(self.change_rates[-5:])))
                
                break
            old_value = np.mean(loss)

    def eval_trajs(self, trajs):
        num_trajs = trajs.shape[0]
        tt = trajs.shape[1]
#         r = np.zeros(num_trajs)
        gammas = np.array([self.gamma ** t for t in range(tt)]).reshape(1, -1)
        r = self.reward(trajs[:,:,0], trajs[:,:,1])
        r = np.sum(r*gammas, axis=1)
#         for t in range(tt):
#             # r += self.reward(trajs[:,t,0], trajs[:,t,1])
#             r += self.reward(trajs[:,t,0], trajs[:,t,1]) * self.gamma**t

        return r

    def eval_policy(self, num_trajs, tt, pm, type = 'old'):
#         print('eval_policy method{}'.format(type))
        if type == 'old':
            trajs = self.sample_trajectory(num_trajs, tt, pm)
        if type == 'new':
            trajs = self.sample_trajectory(num_trajs, tt)
        return self.eval_trajs(trajs)

class  ToyPolicy(tf.keras.Model):
    def __init__(self, simple, init_weight, init_lamb, state_dim =1, action_dim = 2, h_dims=2):
        self.num_A = action_dim
        self.S_dims = state_dim
        self.hidden_dims = h_dims
        self.simple = simple

#         self.input_shape1 = [self.S_dims, self.hidden_dims]
#         self.input_shape2 = [self.hidden_dims, self.num_A]
        if self.simple == True:
            self.input_shape1 = [self.S_dims, self.num_A]
            w1 = tf.cast(init_weight[0],tf.float64)
            self.w1 = tf.Variable(w1, name="w1")
            self.b1 = tf.Variable(tf.zeros(self.input_shape1[1], tf.float64), name="b1")
        else:
            self.input_shape1 = [self.S_dims, self.hidden_dims]
            self.input_shape2 = [self.hidden_dims, self.num_A]
#             self.w1 = self.xavier_var_creator(self.input_shape1, name="w1")
#             self.b1 = tf.Variable(tf.zeros(self.input_shape1[1], tf.float64), name="b1")
# #         self.w2 = self.xavier_var_creator(self.input_shape2, name="w2")
# #         self.b2 = tf.Variable(tf.zeros(self.input_shape2[1], tf.float64), name="b2")
   
            w1 = tf.cast(init_weight[0],tf.float64)
            w2 = tf.cast(init_weight[1],tf.float64)
            self.w1 = tf.Variable(w1, name="w1")
            self.b1 = tf.Variable(tf.zeros(self.input_shape1[1], tf.float64), name="b1")
            self.w2 = tf.Variable(w2, name="w2")
            self.b2 = tf.Variable(tf.zeros(self.input_shape2[1], tf.float64), name="b2")

        self.loglam = tf.Variable(init_lamb*tf.ones(shape=[1,1], dtype=tf.dtypes.float64), name = "loglam")

    def xavier_var_creator(self, input_shape, name="w3"):
        xavier_stddev = 1.0 / tf.sqrt(input_shape[0] / 2.0)
        init = tf.random.normal(shape=input_shape, mean=0.0, stddev=xavier_stddev)
        init = tf.cast(init, tf.float64)
        var = tf.Variable(init, shape=tf.TensorShape(input_shape), trainable=True, name=name)
        return var

    def get_A_prob(self, states, actions=None, multi_dim=True, to_numpy=True):
        """ probability for one action / prob matrix """
        if multi_dim and len(states) > 1:
            states = np.expand_dims(states,axis=-1)
            pre_dims = list(states.shape)
            pre_dims[-1] = self.num_A
            states = states.reshape(-1, self.S_dims)

        z = tf.cast(states, tf.float64)
        """ can specify sparse here
        a_is_sparse = True
        https://www.tensorflow.org/api_docs/python/tf/linalg/matmul
        """
        if self.simple == True:
            probs = tf.nn.softmax(tf.matmul(z,self.w1)+self.b1)
        else:
            h1 = tf.nn.relu(tf.matmul(z, self.w1) + self.b1)
            logit = tf.matmul(h1, self.w2) + self.b2
            probs = tf.nn.softmax(logit)

        if actions is None:
            if multi_dim and len(states) > 1:
                if to_numpy:
                    return probs.numpy().reshape(pre_dims)
                else:
                    return tf.reshape(probs,pre_dims)
            else:
                return probs
        else:
            if to_numpy:
                return probs.numpy()[range(len(actions)), actions]
            else:
                r = probs.numpy()[range(len(actions)), actions]
                return tf.cast(r,tf.float64)



    def sample_A(self, states):

        # np.random.seed(self.seed)
        # self.seed += 1
        A_prob = self.get_A_prob(states) # , to_numpy = to_numpy
        try:
            A_prob = A_prob.numpy()
        except:
            pass
        sample_As = (A_prob.cumsum(1) > np.random.rand(A_prob.shape[0])[:,None]).argmax(1)

        return sample_As