import numpy as np
from _util import *


class simple_generator():
    def __init__(self,gamma,s_min=-5,s_max=5,sigma=0.25):
        print('new_env')
        self.S_dims = 10
        self.num_A = 2

        self.behav = [0.5,0.5]
        self.gamma = gamma
        self.sigma = sigma

        self.s_min = s_min
        self.s_max = s_max
        
    def reward(self, state,action):
        N,T = state.shape[:2]
        s = state[:,:-1]
        s_ = state[:,1:]
        a = action[:,:-1]
        r = s_[:,:,0]*2 + s_[:,:,1] - 1/4*(2*a-1)
        
        return r
    
    def step(self,state,action):

        n = state.shape[0]
        e = np.random.randn(2*n)*self.sigma  #sigma modified 2022-1-23 from 0.25 to 1
        # print('sigma  vl is {}'.format(self.sigma))
#         e = np.zeros(2*n)
        s_ = np.random.randn(n,self.S_dims)
#         s_[:,0] = 0.75*state[:,0] * (2*action-1) + 0.25*state[:,0]*state[:,1] + e[:n]
#         s_[:,1] = 0.75*state[:,1] * (1-2*action) + 0.25*state[:,0]*state[:,1] + e[n:]
        
        s_[:,0] = 0.75*state[:,0] * (2*action-1) + 0.25*state[:,1] + e[:n]
        s_[:,1] = 0.75*state[:,1] * (1-2*action) + 0.25*state[:,0] + e[n:]
        
        
        return s_
    


        
    def sample_behav_traj(self, seed, N, T, burn_in):
        np.random.seed(seed)
        S = np.random.randn(N,self.S_dims)
        all_S = np.zeros((N,T+burn_in+1,self.S_dims))
        all_A = np.random.choice(self.num_A,size = N*(T+burn_in+1),p=self.behav).reshape(N,T+burn_in+1).astype(np.float)

        for t in range(T+burn_in+1):
            all_S[:,t] = S
            S = self.step(S,all_A[:,t])
        r = self.reward(all_S,all_A)
        burnin_S = all_S[:,burn_in:T+burn_in+1]
        burnin_A = all_A[:,burn_in:T+burn_in+1]
        burnin_r =  r[:,burn_in:T+burn_in]
        trajs = [[] for i in range(N)]
        for t in range(T):
            for i in range(N):
                trajs[i].append([burnin_S[i,t],burnin_A[i,t],burnin_r[i,t],burnin_S[i,t+1]])        
        return trajs

#     def sample_policy_traj(self, pi, seed, NN, TT, init_sampler):
#         np.random.seed(seed)
#         S = init_sampler.sample(batch_size=NN)
#         all_S = np.zeros((NN,TT,self.S_dims))
#         all_A = np.zeros((NN,TT))

#         for t in range(TT):
#             all_S[:,t] = S
#             S_norm = (S - self.s_min)/(self.s_max-self.s_min)
#             A = pi.sample_A(S_norm).reshape(-1)
#             all_A[:,t] = A
#             S = self.step(S,A)
#         r = self.reward(all_S,all_A)
#         gene_S = all_S.reshape(-1,self.S_dims)
#         return gene_S
    def sample_S_4VE(self, pi, seed, NN, TT, init_sampler):
        np.random.seed(seed)
        S = init_sampler.sample(batch_size=NN)
        all_S = np.zeros((NN,TT,self.S_dims))
        all_A = np.zeros((NN,TT))

        for t in range(TT):
            all_S[:,t] = S
            S_norm = (S - self.s_min)/(self.s_max-self.s_min)
            A = pi.sample_A(S_norm).reshape(-1)
            all_A[:,t] = A
            S = self.step(S,A)
        gene_S = all_S.reshape(-1,self.S_dims)
        return gene_S    
    def sample_cond_S_4VE(self,states,actions,next_states, pi, M, TT,seed):
        np.random.seed(seed)
        
        TN = states.shape[0]
        
        S = row_repeat(states,M,full_block=True)
        A = np.tile(actions,M)
        gene_S = np.zeros((M*TN,TT,self.S_dims))
        all_A = np.zeros((M*TN,TT))
        gene_S[:,0] = S
        all_A[:,0] = A
        
        for t in range(TT-1):
            S = self.step(gene_S[:,t],all_A[:,t])
            
            gene_S[:,t+1] = S
            S_norm = (S - self.s_min)/(self.s_max-self.s_min)
            A = pi.sample_A(S_norm).reshape(-1)
            all_A[:,t+1] = A
            
        
        gene_S = gene_S.reshape(TN, M, TT, self.S_dims,order='F') 
        ########################
        S = row_repeat(next_states,M,full_block=True)
        S_norm = (S - self.s_min)/(self.s_max-self.s_min)
        A = pi.sample_A(S_norm).reshape(-1)
        gene_S_star = np.zeros((M*TN,TT,self.S_dims))
        all_A = np.zeros((M*TN,TT))
        
        gene_S_star[:,0] = S
        all_A[:,0] = A

        for t in range(TT-1):
            S = self.step(gene_S_star[:,t],all_A[:,t])
            
            gene_S_star[:,t+1] = S
            S_norm = (S - self.s_min)/(self.s_max-self.s_min)
            A = pi.sample_A(S_norm).reshape(-1)
            all_A[:,t+1] = A
        
        gene_S_star = gene_S_star.reshape(TN, M, TT, self.S_dims,order='F')     
        return [gene_S,gene_S_star]
    def eval_trajs(self, trajs):
        r = np.array([[x[2] for x in traj] for traj in trajs])
        N,T = r.shape
        gammas = np.repeat(np.array([self.gamma**i for i in range(T)]).reshape(1,-1),N,axis=0)
        value = np.mean(np.sum(r*gammas,axis=1))
        
        return value 
    def eval_policy(self, pi, init_sampler,seed):
        seed = seed
        NN = 20000
        TT = 100
        pi.seed = seed    #modified by wjn 5-23
        np.random.seed(seed)
        S = init_sampler.sample(batch_size=NN)*(self.s_max-self.s_min) + self.s_min #modified by wjn 6-3
        all_S = np.zeros((NN,TT,self.S_dims))
        all_A = np.zeros((NN,TT))

        for t in range(TT):
            all_S[:,t] = S
            S_norm = (S - self.s_min)/(self.s_max-self.s_min)
            A = pi.sample_A(S_norm).reshape(-1)
            all_A[:,t] = A
            S = self.step(S,A)
        r = self.reward(all_S,all_A)
        N,T = r.shape
        gammas = np.repeat(np.array([self.gamma**i for i in range(T)]).reshape(1,-1),N,axis=0)
        value = np.mean(np.sum(r*gammas,axis=1))
        return value


    
class vl_env():
    def __init__(self):

        self.N = N
        self.T = T
        
        self.curr_state = self.reset()
    
    
    def reset(self):
        self.curr_state = np.random.randn(self.N,2)
        
        return self.curr_state
       
    def rew(self, state, action, next_state):
        r = next_state[:,0]*2 + next_state[:,1] - 1/4*(2*action -1)
        return r

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
        
    