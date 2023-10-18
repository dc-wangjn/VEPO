from random import sample
import numpy as np
import time
from sklearn.kernel_approximation import RBFSampler


def sigmoid(x):
    return 1/(1+np.exp(-x))

class FQI_module():
    def __init__(self, gamma, S_dims, feature_dim):
        self.S_dims = S_dims
        self.gamma = gamma
        self.num_A = 2
        self.feature_dim = feature_dim
        self.rbf_feature = RBFSampler(gamma=1, n_components=feature_dim,random_state = 10)
  
    def reward(self, state,action):
        N,T = state.shape[:2]
        s = state[:,:-1]
        s_ = state[:,1:]
        a = action[:,:-1]
        r = s_[:,:,0]*2 + s_[:,:,1] - 1/4*(2*a-1)
        
        return r

    def train(self, iters, trajs):
        states = trajs[:,:-1,:self.S_dims].reshape(-1,self.S_dims)
        action = trajs[:,:-1,-1].reshape(-1).astype(np.int)
        next_states = trajs[:,1:,:self.S_dims].reshape(-1,self.S_dims)
        reward = self.reward(trajs[:,:,:self.S_dims],trajs[:,:,-1]).reshape(-1)

        ss = self.rbf_feature.fit_transform(states)
        ss_ = self.rbf_feature.fit_transform(next_states)
        aa = np.eye(self.num_A)[action]
        xi = np.concatenate([np.kron(y,x).reshape(1,-1) for x,y in zip(ss,aa)])

        sigma = np.dot(xi.T,xi)+ 0.000001 * np.eye(self.feature_dim*self.num_A)
        sigma_inv = np.linalg.inv(sigma)
        tmp_mat =  np.dot(sigma_inv,xi.T)

        beta = np.dot(tmp_mat,reward[:,None]).reshape(self.feature_dim,-1)

        for i in range(iters):
            tgt = reward +  self.gamma * np.max(np.dot(ss_,beta),axis=1)
            beta = np.dot(tmp_mat,tgt[:,None]).reshape(self.feature_dim,-1)


        self.beta = beta

    def get_prob(self, states):

        p_a1 = self.sample_A(states)
        prob = np.eye(self.num_A)[p_a1]
        return prob
        
    def sample_A(self,states):
        shape = list(states.shape[:-1])
        sdims = states.shape[-1]
        states = states.reshape(-1,sdims)

        s = self.rbf_feature.fit_transform(states)
        a = np.argmax(np.dot(s,self.beta),axis=1).reshape(shape)
        return a


class V_module():
    def __init__(self, beta, pi, feature_dim):
        self.beta = beta 
        self.pi = pi
        self.feature_dim = feature_dim
        self.rbf_feature = RBFSampler(gamma=1, n_components=feature_dim,random_state = 10)

    def get_q(self, states):
        s = self.rbf_feature.fit_transform(states)
        b = self.beta.reshape(self.feature_dim,-1)
        return np.dot(s,b)

    def get_v(self, states):
        q = self.get_q(states)
        prob = self.pi.get_prob(states)
        v = np.sum(q*prob,axis=-1)
        return v

    def get_a(self,states):
        shape = list(states.shape[:-1])
        sdims = states.shape[-1]
        adim = 2
        states = states.reshape(-1,sdims)
        q = self.get_q(states)
        v = self.get_v(states)
        a = q - v[:,None]
        a = a.reshape(shape+[adim])
        return a 

class CO_module():
    def __init__(self, beta, pi, s0, feature_dim):
        self.s0 = s0
        self.beta = beta 
        self.pi = pi
        self.feature_dim = feature_dim
        self.rbf_feature = RBFSampler(gamma=1, n_components=feature_dim,random_state = 10)
    
    def construct_basis(self,s,a,s_,a_):
        n1 = s.shape[0]
        n2 = s_.shape[0]
        a = a.reshape(-1).astype(np.int)
        a_ = a_.reshape(-1).astype(np.int)

        sr = np.repeat(s,n2,axis=0)
        st_ = np.tile(s_,(n1,1))
        ar = np.repeat(a,n2)
        at_ = np.tile(a_,n1) 

        ssr = self.rbf_feature.fit_transform(sr)
        sst_ = self.rbf_feature.fit_transform(st_)

        aar = np.eye(2)[ar]
        aat_ = np.eye(2)[at_]

        ssrt_ = np.concatenate([ssr,sst_],axis=-1)
        ssrtar = np.concatenate([np.kron(y,x).reshape(1,-1) for x,y in zip(ssrt_,aar)])
        ssrtarat = np.concatenate([np.kron(y,x).reshape(1,-1) for x,y in zip(ssrtar,aat_)])
        return ssrtarat
    
    def get_cond_omega(self, s,a,s_,a_):
        '''
        input: s123,a123,s_123,a_123
        output: cond_omega(s1a1s_1a_1,s1a1s_2a_2,s1a1s_3a_3,s2a2s_1a_1,...)
        out shape: (s.shape[0] *s_.shape[0], )
        '''
        basis = self.construct_basis(s,a,s_,a_)
        n1 = s.shape[0]
        n2 = s_.shape[0]
        cond_omega = np.maximum(np.dot(basis,self.beta.reshape(-1)[:,None]),0.001)
        
        cond_omega = cond_omega.reshape(n1,n2)
        mean_omega = np.mean(cond_omega,axis=0).reshape(1,-1) + 0.000001
        cond_omega = cond_omega/mean_omega
        return cond_omega.reshape(-1)

    def get_omega(self,s,a):
        n1 = s.shape[0]
        a0 = self.pi.sample_A(self.s0)
        cond_omega = self.get_cond_omega(s,a,self.s0,a0)
        cond = cond_omega.reshape(n1,-1)
        omega = np.mean(cond,axis=-1)
        omega = omega/np.mean(omega)
        return omega

class Transition():
    def __init__(self, beta, feature_dim):

        self.beta = beta 
        self.feature_dim = feature_dim
        self.num_A = 2 
        self.rbf_feature = RBFSampler(gamma=1, n_components=feature_dim,random_state = 10)

    def step(self, states, action):
        # print('using estimated transition....')
        # print('s is ')
        # print(states[:10])
        shape = list(states.shape[:-1])
        sdims = states.shape[-1]
        states = states.reshape(-1,sdims)
        action = action.astype(np.int)
        ss = self.rbf_feature.fit_transform(states)
        aa = np.eye(self.num_A)[action]
        xi = np.concatenate([np.kron(y,x).reshape(1,-1) for x,y in zip(ss,aa)])
        ss_ = np.dot(xi,self.beta)
        ss_ = ss_.reshape(shape+[sdims])
        # print('s_ is {}'.format(ss_))
        
        return ss_


        
class Policy():
    def __init__(self, parms, feature_dim):
        self.parms = parms.reshape(-1)
        self.feature_dim = feature_dim
        self.rbf_feature = RBFSampler(gamma=1, n_components=feature_dim,random_state = 10)
    def get_prob(self, states):
        shape = list(states.shape[:-1])
        sdims = states.shape[-1]
        adim = 2
        states = states.reshape(-1,sdims)

        s = self.rbf_feature.fit_transform(states)
        p_a1 = sigmoid(np.dot(s,self.parms[:,None]))
        p_a2 = 1 - p_a1
        prob = np.concatenate([p_a1,p_a2],axis=-1)
        prob = prob.reshape(shape+[adim])
        return prob
    def sample_A(self,states):
        prob = self.get_prob(states)
        sample_As = (prob.cumsum(-1) > np.expand_dims(np.random.rand(*prob.shape[:-1]),-1)).argmax(-1)
        return sample_As

class Random_Policy():

    def sample_A(self,states):
        shape = list(states.shape[:-1])
        return np.random.randint(0,2,shape)
        






