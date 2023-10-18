#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
TODO: allow online interating. take a step.
"""

""" Usage
1. evaluation
    1. (multi-seeds) simulate a dataset with the behaviour policy, OPE the target policy.
    2. evaluate in the simulator (multi-seeds)
2. offline RL
    1. (multi-seeds) simulate a dataset with the behaviour policy, learn a policy (ours or competing), and evaluate in the simulator (multi-seeds)
"""
# import os, sys
# package_path = os.path.dirname(os.path.abspath(os.getcwd()))
# sys.path.insert(0, package_path + "/test_func")
# from _util_TRPO import *
from _util import *
import operator
########################################################################################################################################################################################################################################################################################################################################################################################


class OhioSimulator():
    def __init__(self,s_min=0,s_max=300, sd_G = 3, T = 20, N = 10):
        # the following parameters will not change with the LM fitting
        self.CONST = 39.03
        #####
        self.init_u_G = 162  
        self.init_sd_G = 60
        #####
        self.p_D, self.u_D, self.sd_D = 0.17, 44.4, 35.5
        self.p_E, self.u_E, self.sd_E = 0.05, 4.9, 1.04
        self.p_A = [0.805, 0.084, 0.072, 0.029, 0.010] # new discritization
        
        self.range_a = [0, 1, 2, 3, 4]
        # left to right: t-4, .. , t-1
        self.coefficients = [-0.008     ,  0.106     , -0.481     ,  1.171  # glucose
          , 0.008     ,  -0.004     ,  0.08      ,  0.23      # diet
          , 0.009     , -1.542     , 3.097     , -3.489  # exercise
          , -0.30402253, -2.02343638, -0.3310525 , -0.43941028] # action
        self.tran_mat = np.expand_dims(arr(self.coefficients), 0)
        self.sd_G = sd_G
        self.T, self.N = T, N
        self.seed = 42
        self.true_lag = 4
        
        self.s_min = s_min
        self.s_max = s_max
        
    def Glucose2Reward(self, gls):
        low_gl, high_gl = 80, 140
        rewards = np.select([gls >= high_gl, gls <= low_gl, np.multiply(low_gl < gls, gls < high_gl)]
          , [-(gls - high_gl) ** 1.35 / 30, -(low_gl - gls) ** 2 / 30, 0])
        return rewards
############################################################################################################################################
############################################################################################################################################

    def init_MDPs(self, TT = None, NN = None, seed = 0): 
        """ Randomly initialize 
        1. G_t [0,..., 4]
        1. the other state variable: random.
        2. errors for G_t
        3. when to take how many diets/exercises [matters?]
        
        where T varies, seed is diff; dominated by init several states, and hence values are not monotone.
        """
        np.random.seed(seed)
        if TT is None and NN is None:
            T, N = self.T, self.N
        else:
            T,N = TT,NN
        obs = np.zeros((3, T, N))  # [Gi, D, Ex]
        e_D = abs(rnorm(self.u_D, self.sd_D, T * N))
        e_E = abs(rnorm(self.u_E, self.sd_E, T * N))
        e_G = rnorm(0, self.sd_G, T * N).reshape((T, N))

        obs[0, :4, :] = rnorm(self.init_u_G, self.init_sd_G, 4 * N).reshape(4, N)
        obs[1, :, :] = (rbin(1, self.p_D, T * N) * e_D).reshape((T, N))
        obs[2, :, :] = (rbin(1, self.p_E, T * N) * e_E).reshape((T, N))
        return obs, e_G
    ##################################################################################################################################################################
    def conc_single(self, obs, actions, t):
        # obs = [3, T]
        # actions = [T]
        As = actions[(t - self.true_lag + 1):t]
        S = obs[:, (t - self.true_lag + 1):(t + 1)]
        s = S.ravel(order='C')
        s = np.append(s, As)
        return s

    def reset(self, T = 1000):
        self.seed += 1
        if self.seed > 1e4:
            self.seed = 42
        self.T = T
        np.random.seed(self.seed)
        ### Initialization
        self.obs = np.zeros((3, T)) 
        self.e_D = abs(rnorm(self.u_D, self.sd_D, T))
        self.e_E = abs(rnorm(self.u_E, self.sd_E, T))
        self.e_G = rnorm(0, self.sd_G, T)
        self.obs[0, :4] = rnorm(self.init_u_G, self.init_sd_G, 4)
        self.obs[1, :] = rbin(1, self.p_D, T) * self.e_D
        self.obs[2, :] = rbin(1, self.p_E, T) * self.e_E
        self.t = 3
        self.actions = zeros(T)
        self.actions[:3] = np.random.choice(range(len(self.p_A)), size = 3, p = self.p_A)
        states = self.conc_single(obs = self.obs, actions = self.actions, t = self.t)
        return states
    
    def online_step(self, action):
        self.actions[self.t] = action
        # 1. transition
        states = self.conc_single(obs = self.obs, actions = self.actions, t = self.t)
        SA = np.append(states, action)
        self.t += 1
        next_G = self.CONST + self.tran_mat.dot(SA) + self.e_G[self.t]
        # 2. store
        self.obs[0, self.t] = next_G
        # 3. return 
        observation_ = self.conc_single(obs = self.obs, actions = self.actions, t = self.t)
        done = (self.t == (self.T - 1))
        reward = self.Glucose2Reward(next_G)
        reward += (randn(1) / 1e4)
        reward = np.squeeze(reward)
#         s = np.hstack([obs[(self.t - self.true_lag + 1):self.t, :], actions[(self.t - self.true_lag + 1):self.t]]).ravel(order='C')
#         s = np.append(s, obs[self.t, :].ravel())
        return observation_, reward, done

############################################################################################################################################
############################################################################################################################################
    def step(self, states = None, errors = None):
        return np.array(self.CONST).reshape((1, 1)) + self.tran_mat.dot(states) + errors
    
    def simu_one_seed(self, seed = 42):
        """ Simulate N patient trajectories with length T, calibrated from the Ohio dataset.
        Returns:
            trajs = [traj], where traj [[S, A, R, SS]]
        """
        #########
        T, N = self.T, self.N
        np.random.seed(seed)
        # Initialization
        obs, e_G = self.init_MDPs(seed = seed)
        actions = np.random.choice(range(len(self.p_A)), size = T * N, p = self.p_A).reshape((T, N))
            
        ######### Transition
        for t in range(4, T):
            states = self.concatenate_useful_obs(obs = obs, actions = actions, t = t)
            obs[0, t, :] = self.step(states = states, errors = np.array([e_G[t, :]]))
        actions = actions.astype(float)
        
        ######### Collection: obs = (3, T, N), actions = (T, N)
        # conc_SA_2_state(obs, actions, t, multiple_N = False, J = 4) 
        trajs = [[] for i in range(N)]
        for t in range(4, T - 1): 
            conc_s = self.conc_SA_2_state(obs, actions, t, multiple_N = True, J = 4) # (dim_with_J, N)
            conc_ss = self.conc_SA_2_state(obs, actions, t + 1, multiple_N = True, J = 4)# (dim_with_J, N)
            As = actions[t, :]
            Rs = self.Glucose2Reward(conc_ss[-3, :])
            for i in range(N):
                trajs[i].append([conc_s[:, i], As[i], Rs[i], conc_ss[:, i]])
        return trajs
        
    def simu_multi_seeds(self, M, parallel = True):
        if parallel:
            return parmap(self.simu_one_seed, range(M))
        else:
            return [self.simu_one_seed(i) for i in range(M)]
############################################################################################################################################
############################################################################################################################################
    def cum_r(self, rewards, gamma):
        """ rewards -> culmulative reward
        """
        return sum(map(operator.mul, [gamma ** j for j in range(len(rewards))], rewards))

    def eval_trajs(self, trajs = None, gamma = .8):        
        def get_rew(i): 
            glucoses = arr([item[0][-3] for item in trajs[i]])
            rewards = np.roll(self.Glucose2Reward(glucoses), shift = -1).reshape(-1, 1)
            discounted_value = np.round(self.cum_r(rewards, gamma), 3)[0]
            average_value = round(np.mean(rewards))
            return [rewards, discounted_value, average_value]
        res = parmap(get_rew, range(len(trajs)))
        discounted_values = [a[1] for a in res]
        average_values = [a[2] for a in res]
        return discounted_values, average_values

    def eval_policy(self, pi, gamma = 1, seed = 42, return_value = True):
        """ Evaluate the value of a policy in simulation.
        sample the first four time points，and then begin to follow the policy and collect rewards
        transform into matrix so that linear transition is easier
        Concatenate data for training and evaluation 
        """
        pi.seed = seed
        np.random.seed(seed)
        ##############################
        obs, e_G = self.init_MDPs(seed = seed) # obs = [3, T, N]
        actions = np.zeros((self.T, self.N)) # store previous actions+
        actions[:4, :] = np.random.choice(range(len(self.p_A)), size = 4 * self.N, p = self.p_A).reshape((4, self.N))
        self.init = [obs.copy(), e_G.copy(), actions[:4, :].copy()]
        ##############################
        curr_time = now()
        for t in range(4, self.T): 
            # next observations: based on ...,t-1, to decide t.
            states = self.concatenate_useful_obs(obs = obs, actions = actions, t = t)
            obs[0, t, :] = self.step(states = states, errors = np.array([e_G[t, :]]))

            # choose actions based on status. obs = [3, T, N]      
            S = self.conc_SA_2_state(obs, actions, t, multiple_N = True).T #  N * dim
            if t == 4:
                self.init_state = S.copy()
            S_norm = (S - self.s_min)/(self.s_max-self.s_min)
            A_t = pi.sample_A(S_norm) # s [N * dx] -> actions [N * 1] -> 1 * N
            actions[t, :] = np.squeeze(A_t)
        ##############################
        if return_value:
            discounted_values, average_values = self.cal_reward(obs, gamma)
            return discounted_values, average_values # len-N
        else:
            ######### Collection: obs = (3, T, N), actions = (T, N)
            # conc_SA_2_state(obs, actions, t, multiple_N = False, J = 4) 
            trajs = [[] for i in range(self.N)]
            for t in range(4, self.T - 1): 
                conc_s = self.conc_SA_2_state(obs, actions, t, multiple_N = True, J = 4) # (dim_with_J, N)
                conc_ss = self.conc_SA_2_state(obs, actions, t + 1, multiple_N = True, J = 4)# (dim_with_J, N)
                As = actions[t, :]
                Rs = self.Glucose2Reward(conc_ss[-3, :])
                for i in range(self.N):
                    trajs[i].append([conc_s[:, i], As[i], Rs[i], conc_ss[:, i]])
            return trajs
        
    def generate_mixture_policy_trajectory(self, pi, epsilon, gamma = 1, seed = 42):
        """ Evaluate the value of a policy in simulation.
        sample the first four time points，and then begin to follow the policy and collect rewards
        transform into matrix so that linear transition is easier
        Concatenate data for training and evaluation 
        """
        pi.seed = seed
        np.random.seed(seed)
        ##############################
        obs, e_G = self.init_MDPs(seed = seed) # obs = [3, T, N]
        actions = np.zeros((self.T, self.N)) # store previous actions+
        actions[:4, :] = np.random.choice(range(len(self.p_A)), size = 4 * self.N, p = self.p_A).reshape((4, self.N))
        self.init = [obs.copy(), e_G.copy(), actions[:4, :].copy()]
        ##############################
      
        curr_time = now()
        for t in range(4, self.T): 
            # next observations: based on ...,t-1, to decide t.
            states = self.concatenate_useful_obs(obs = obs, actions = actions, t = t)
            obs[0, t, :] = self.step(states = states, errors = np.array([e_G[t, :]]))

            # choose actions based on status. obs = [3, T, N]      
            S = self.conc_SA_2_state(obs, actions, t, multiple_N = True).T #  N * dim
            if t == 4:
                self.init_state = S.copy()
                
            
            if np.random.random() < epsilon:   
                S_norm = (S - self.s_min)/(self.s_max-self.s_min)
                A_t = pi.sample_A(S_norm) # s [N * dx] -> actions [N * 1] -> 1 * N
#                 A_t = pi.sample_A(S) # s [N * dx] -> actions [N * 1] -> 1 * N
            else:
                A_t =  np.random.choice(range(len(self.p_A)), size = self.N, p = self.p_A)
            actions[t, :] = np.squeeze(A_t)
        discounted_values, average_values = self.cal_reward(obs, gamma)
            
        ##############################

        trajs = [[] for i in range(self.N)]
        for t in range(4, self.T - 1): 
            conc_s = self.conc_SA_2_state(obs, actions, t, multiple_N = True, J = 4) # (dim_with_J, N)
            conc_ss = self.conc_SA_2_state(obs, actions, t + 1, multiple_N = True, J = 4)# (dim_with_J, N)
            As = actions[t, :]
            Rs = self.Glucose2Reward(conc_ss[-3, :])
            for i in range(self.N):
                trajs[i].append([conc_s[:, i], As[i], Rs[i], conc_ss[:, i]])
        return trajs,discounted_values, average_values

    
##############################TODO######################################################    
    
    def sample_S_4VE(self, pi, NN, TT,  seed = 42):
        """ Evaluate the value of a policy in simulation.
        sample the first four time points，and then begin to follow the policy and collect rewards
        transform into matrix so that linear transition is easier
        Concatenate data for training and evaluation 
        """
        pi.seed = seed
        np.random.seed(seed)
        ##############################
        TT_trajs = TT+4
        obs, e_G = self.init_MDPs(TT = TT_trajs, NN = NN, seed = seed) # obs = [3, T, N]
        actions = np.zeros((TT_trajs, NN)) # store previous actions+
        S0 = self.conc_SA_2_state(obs, actions, 4, multiple_N = True).T
        dim_S = S0.shape[1]
        gene_S = np.zeros((TT,NN,dim_S))
        
        actions[:4, :] = np.random.choice(range(len(self.p_A)), size = 4 * NN, p = self.p_A).reshape((4, NN))
        self.init = [obs.copy(), e_G.copy(), actions[:4, :].copy()]
        ##############################
        curr_time = now()
        for t in range(4, TT_trajs): 
            # next observations: based on ...,t-1, to decide t.
            states = self.concatenate_useful_obs(obs = obs, actions = actions, t = t)
            obs[0, t, :] = self.step(states = states, errors = np.array([e_G[t, :]]))

            # choose actions based on status. obs = [3, T, N]      
            S = self.conc_SA_2_state(obs, actions, t-1, multiple_N = True).T #  N * dim
            gene_S[t-4,:,:] = S
            if t == 4:
                self.init_state = S.copy()
                
                
            S_norm = (S - self.s_min)/(self.s_max-self.s_min)
            A_t = pi.sample_A(S_norm) # s [N * dx] -> actions [N * 1] -> 1 * N
#             A_t = pi.sample_A(S) # s [N * dx] -> actions [N * 1] -> 1 * N
            actions[t, :] = np.squeeze(A_t)
        ##############################
        gene_S = gene_S.transpose(1,0,2)
        gene_S = gene_S.reshape(-1, dim_S)  # stack gene_S; (TTNN, dim_S)
        return gene_S        

    def sample_cond_S_4VE(self, states, actions, next_states, pi, M, TT, seed = 42):

        pi.seed = seed
        np.random.seed(seed)
        ##############################
        TT_trajs = TT+4
        dim_S = states.shape[1]
        TN = states.shape[0]
        obs, e_G = self.init_MDPs(TT = TT_trajs, NN = TN*M, seed = seed) # obs = [3, T, N]
        A = np.zeros((TT_trajs, M*TN)) # store previous actions+
#         S0 = self.conc_SA_2_state(obs, actions, 4, multiple_N = True).T
        
        gene_S = np.zeros((TT,TN*M,dim_S))
        #####################################
        S0 = row_repeat(states,M,full_block=True)
        
        A0 = np.tile(actions,M)
        J = int(dim_S/4)
        for i in range(J):
            obs[:,i] = S0[:,(i*4):(i*4)+3].transpose()
            A[i,:] = S0[:,i*4+3]
        
        obs[:,3] = S0[:,12:].transpose()
        A[3, :] = A0
        self.init = [obs.copy(), e_G.copy(), A[:4, :].copy()]
        ##############################
        for t in range(4, TT_trajs): 
            # next observations: based on ...,t-1, to decide t.
            states = self.concatenate_useful_obs(obs = obs, actions = A, t = t)
            obs[0, t, :] = self.step(states = states, errors = np.array([e_G[t, :]]))

            # choose actions based on status. obs = [3, T, N]      
            S = self.conc_SA_2_state(obs, A, t-1, multiple_N = True).T #  N * dim
            gene_S[t-4,:,:] = S
            if t == 4:
                self.init_state = S.copy()
                
                
            S_norm = (S - self.s_min)/(self.s_max-self.s_min)
            A_t = pi.sample_A(S_norm) # s [N * dx] -> actions [N * 1] -> 1 * N
#             A_t = pi.sample_A(S) # s [N * dx] -> actions [N * 1] -> 1 * N
            A[t, :] = np.squeeze(A_t)
        ##############################
        
        gene_S = gene_S.transpose(1,0,2)
        gene_S = gene_S.reshape(TN, M, TT, dim_S,order='F') 
        #########################################################
        obs, e_G = self.init_MDPs(TT = TT_trajs, NN = TN*M, seed = seed) # obs = [3, T, N]
        A = np.zeros((TT_trajs, M*TN)) # store previous actions+
        gene_S_star = np.zeros((TT,TN*M,dim_S))
        S0 = row_repeat(next_states,M,full_block=True)
        S0_norm = (S0 - self.s_min)/(self.s_max-self.s_min)
        A0 = pi.sample_A(S0_norm) # s [N * dx] -> actions [N * 1] -> 1 * N
#         A0 = pi.sample_A(S0)
        
        J = int(dim_S/4)
        for i in range(J):
            obs[:,i] = S0[:,(i*4):(i*4)+3].transpose()
            A[i,:] = S0[:,i*4+3]
        
        obs[:,3] = S0[:,12:].transpose()
        A[3, :] = np.squeeze(A0)
        self.init = [obs.copy(), e_G.copy(), A[:4, :].copy()]
        ##############################
        for t in range(4, TT_trajs): 
            # next observations: based on ...,t-1, to decide t.
            states = self.concatenate_useful_obs(obs = obs, actions = A, t = t)
            obs[0, t, :] = self.step(states = states, errors = np.array([e_G[t, :]]))

            # choose actions based on status. obs = [3, T, N]      
            S = self.conc_SA_2_state(obs, A, t-1, multiple_N = True).T #  N * dim
            gene_S_star[t-4,:,:] = S
            if t == 4:
                self.init_state = S.copy()

            S_norm = (S - self.s_min)/(self.s_max-self.s_min)
            A_t = pi.sample_A(S_norm) # s [N * dx] -> actions [N * 1] -> 1 * N
#             A_t = pi.sample_A(S) # s [N * dx] -> actions [N * 1] -> 1 * N
            A[t, :] = np.squeeze(A_t)
        ##############################  
        gene_S_star = gene_S_star.transpose(1,0,2)
        gene_S_star = gene_S_star.reshape(TN, M, TT, dim_S,order='F') 
        return [gene_S,gene_S_star]       
    
    
    
    def cal_reward(self, obs, gamma):
        all_rewards = self.Glucose2Reward(obs)
        all_rewards = all_rewards[0]
        all_rewards = np.roll(all_rewards[4:], shift = -1, axis = 0)
        gammas = np.expand_dims(arr([gamma ** j for j in range(all_rewards.shape[0])]), 0)
        discounted_values = np.dot(gammas, all_rewards)
        average_values = np.mean(all_rewards, 0)
        return discounted_values, average_values
    #########################################################################################################

############################################################################################################################################
############################################################################################################################################

    def concatenate_useful_obs(self, obs, actions, t, true_lag = 4):
        # (dim_with_J, N)
        r = np.vstack([
            obs[0, (t - true_lag):t, :], obs[1, (t - true_lag):t, :],
            obs[2, (t - true_lag):t, :], actions[(t - true_lag):t, :]])
        return r


    def conc_SA_2_state(self, obs, actions, t, multiple_N = False, J = 4):
        """ to form a lag-J states from history obs and A
        """
        # dim = (3, T, N)
        N = obs.shape[2]
        dim_obs = 3
        s = np.vstack(([
            obs[:, (t - J + 1 ):t, :],
            actions[(t - J + 1):t, :].reshape((1, J - 1, N))])) # extend_dim for first one
        s = s.reshape(((dim_obs + 1) * (J - 1), N), order = 'F')
        obs_0 = obs[:, t, :] # 3 * N
        s = np.vstack([s, obs_0])
        return s # (dim_with_J, N)
    
    
    
    
    def sample(self, states, actions):
        
        trans_mat = np.reshape(np.array(self.coefficients).reshape(4,4),(-1,1),order='F')
        SA = np.hstack([states,actions])
        TN = int(len(SA))
        next_g = SA.dot(trans_mat) +  rnorm(0,self.sd_G,TN).reshape(-1,1) +self.CONST
        
        e_D = abs(rnorm(self.u_D, self.sd_D, TN))
        e_E = abs(rnorm(self.u_E, self.sd_E, TN))
        next_d = (rbin(1, self.p_D, TN) * e_D).reshape(-1,1)
        next_e = (rbin(1, self.p_E, TN) * e_E).reshape(-1,1)
        ns = np.hstack([next_g,next_d,next_e])
        
        next_states = np.hstack([SA[:,4:],ns])
        return next_states
