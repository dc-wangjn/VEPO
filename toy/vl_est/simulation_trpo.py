import numpy as np
import time
from sklearn.kernel_approximation import RBFSampler

from modules import *


class VL_SIMU(object):
    def __init__(self,gamma, feature_dim,delta):

        self.S_dims = 15
        self.num_A = 2

        self.behav = [0.5,0.5]
        self.gamma = gamma
        
        self.rbf_feature = RBFSampler(gamma=1, n_components=feature_dim,random_state = 10)
        self.feature_dim = feature_dim
        self.KL_delta = delta
        
    def reward(self, state,action):
        N,T = state.shape[:2]
        s = state[:,:-1]
        s_ = state[:,1:]
        a = action[:,:-1]
        r = s_[:,:,0]*2 + s_[:,:,1] - 1/4*(2*a-1)
        
        return r

    
    def step(self,state,action):

        n = state.shape[0]
        e = np.random.randn(2*n)*0.25  #sigma modified 2022-1-23 from 0.25 to 1
#         e = np.zeros(2*n)
        s_ = np.random.randn(n,self.S_dims)

        s_[:,0] = 0.75*state[:,0] * (2*action-1) + 0.25*state[:,1] + e[:n]
        s_[:,1] = 0.75*state[:,1] * (1-2*action) + 0.25*state[:,0] + e[n:]

        return s_
    
    def init_sampler(self, num_trajs):
        init_state = np.random.randn(num_trajs,self.S_dims)
        return init_state

    def learn_init_policy(self, num_trajs, tt, iters, init_policy):
        pib = Random_Policy()
        trajs = self.sample_trajectory(num_trajs,tt,pib,None)
        if init_policy == 'FQI':
            init_pi = FQI_module(self.gamma, self.S_dims, self.feature_dim)
            init_pi.train(iters, trajs)
            return init_pi

    def cal_q_function(self, pi, num_trajs, tt):

        pib = Random_Policy()
        trajs = self.sample_trajectory(num_trajs,tt,pib,None)

        state = trajs[:,:-1,:self.S_dims].reshape(-1,self.S_dims)
        action = trajs[:,:-1,-1].reshape(-1).astype(np.int)
        next_state = trajs[:,1:,:self.S_dims].reshape(-1,self.S_dims)
        reward = self.reward(trajs[:,:,:self.S_dims],trajs[:,:,-1]).reshape(-1)


        ss = self.rbf_feature.fit_transform(state)
        ss_ = self.rbf_feature.fit_transform(next_state)

        aa = np.eye(self.num_A)[action]
        
        xi = np.concatenate([np.kron(y,x).reshape(1,-1) for x,y in zip(ss,aa)])

        pi_ = pi.get_prob(next_state) #(n, self.num_A)
        pi_ss_ = np.concatenate([np.kron(y,x).reshape(1,-1) for x,y in zip(ss_,pi_)])

        res = xi - self.gamma*pi_ss_
        
        sigma =  np.mean(np.stack([np.dot(x[:,None],y[None,:]) for x,y in zip(xi,res)]),axis=0) + 0.000001 * np.eye(self.feature_dim*self.num_A)
        # print(sigma)
        sigma_inv = np.linalg.inv(sigma)
        tgt = np.mean(xi * reward[:,None],axis=0)
        
        beta = np.dot(sigma_inv,tgt[:,None])
        q = V_module(beta, pi, self.feature_dim)
        
        return q



    def cal_density_ratio(self, num_trajs,tt, pi):
        pib = Random_Policy()
        trajs = self.sample_trajectory(num_trajs,tt,pib,None)

        s = trajs[:,:-1,:self.S_dims].reshape(-1,self.S_dims)
        a = trajs[:,:-1,-1].reshape(-1).astype(np.int)
        s_ = trajs[:,1:,:self.S_dims].reshape(-1,self.S_dims)


        s0 = trajs[:,0,:self.S_dims].reshape(-1,self.S_dims)

        ss = self.rbf_feature.fit_transform(s)
        ss_ = self.rbf_feature.fit_transform(s_)

        aa = np.eye(self.num_A)[a]

        _s = s[::-1]
        _a = a[::-1]
        _s_ = s_[::-1]

        _ss = self.rbf_feature.fit_transform(_s)
        _aa = np.eye(self.num_A)[_a]

        _ssss = np.concatenate([ss,_ss],axis=-1)
        ssa = np.concatenate([np.kron(y,x).reshape(1,-1) for x,y in zip(_ssss,aa)])
        phi = np.concatenate([np.kron(y,x).reshape(1,-1) for x,y in zip(ssa,_aa)])

        pi_ = pi.get_prob(s_)
        _ssss_ = np.concatenate([_ss,ss_],axis=-1)
        sspi = np.concatenate([np.kron(y,x).reshape(1,-1) for x,y in zip(_ssss_,pi_)])
        u_ = np.concatenate([np.kron(y,x).reshape(1,-1) for x,y in zip(sspi,_aa)])

        res = phi - self.gamma*u_
        sigma =  np.mean(np.stack([np.dot(x[:,None],y[None,:]) for x,y in zip(res,phi)]),axis=0) + 0.000001 * np.eye(self.num_A*self.num_A*2*self.feature_dim)
        sigma_inv = np.linalg.inv(sigma)


        pi0 = pi.get_prob(s0)
        ss0 = self.rbf_feature.fit_transform(s0)
        ss0ss0 = np.concatenate([ss0,ss0],axis=-1)
        ssa0 = np.concatenate([np.kron(y,x).reshape(1,-1) for x,y in zip(ss0ss0,pi0)])
        _phi = np.concatenate([np.kron(y,x).reshape(1,-1) for x,y in zip(ssa0,pi0)])

        tgt = np.mean(_phi,axis=0)*(1-self.gamma)

        beta = np.dot(sigma_inv,tgt[:,None])
        
        omega = CO_module(beta, pi, s0, self.feature_dim) #
        
        return omega


    def cal_transition(self, num_trajs, tt ):

        pib = Random_Policy()
        trajs = self.sample_trajectory(num_trajs,tt,pib,None)

        state = trajs[:,:-1,:self.S_dims].reshape(-1,self.S_dims)
        action = trajs[:,:-1,-1].reshape(-1).astype(np.int)
        next_state = trajs[:,1:,:self.S_dims].reshape(-1,self.S_dims)


        ss = self.rbf_feature.fit_transform(state)
        # ss = state
        aa = np.eye(self.num_A)[action]
        xi = np.concatenate([np.kron(y,x).reshape(1,-1) for x,y in zip(ss,aa)])

        sigma = np.dot(xi.T,xi)+ 0.000001 * np.eye(self.feature_dim*self.num_A)
        sigma_inv = np.linalg.inv(sigma)

        beta = np.dot(np.dot(sigma_inv,xi.T),next_state)

        trans = Transition(beta, self.feature_dim)
        print('original s_ is {}'.format(next_state[:10]))
        est_ss = trans.step(state,action)
        print('est s_ is {}'.format(est_ss[:10]))
        sst = np.sum(np.square(est_ss - next_state))/est_ss.shape[0]
        print('error is {}'.format(sst))
        return trans
        

    def compute_components(self,  pi, cal_nu_nn,cal_nu_mm,cal_nu_tt,M,M2, tt, trajs_behav = None,v_func = None ,omega = None, trans=None):
        '''sholud use test set as trajs_behav instead of the whole trajs'''

        if omega is None:
#             print('calculating density ratio ... ')
            omega, = self.cal_density_ratio(cal_nu_nn,cal_nu_mm,cal_nu_tt, pi)


        # n1*tt1
        state = trajs_behav[:,:-1,:self.S_dims].reshape(-1,self.S_dims)
        action = trajs_behav[:,:-1,-1].reshape(-1).astype(np.int)
        next_state = trajs_behav[:,1:,:self.S_dims].reshape(-1,self.S_dims)
        reward = self.reward(trajs_behav[:,:,:self.S_dims],trajs_behav[:,:,-1]).reshape(-1)

        self.nt = state.shape[0]
        nt = self.nt

        # M,tt2
        gammas = np.array([self.gamma ** t for t in range(tt)]).reshape(1, -1, 1)

        startT = time.time()
#         print('sample init trajs ...')
        trajs_star = self.sample_trajectory(M, tt, pi, trans)
        print('Done. cost {}'.format(time.time()-startT))

        # n1*tt1,m,tt2,2
        print('sample cond trajs ...')
        startT = time.time()
        trajs_cond = self.sample_cond_trajectory(M2, tt,pi,state,action,trans).reshape(nt, M2, tt, self.S_dims+1)
        print('Done. cost {}'.format(time.time() - startT))
    # init_next_state = next_state[:,0].astype(np.int)
        next_a0 = np.zeros(self.nt)
        next_a1 = np.ones(self.nt)
        # (dim_a, n1*tt1,m, tt2, 2)
        print('sample next cond trajs ...')
        startT = time.time()
        trajs_cond_next = np.stack([self.sample_cond_trajectory(M2, tt,pi,next_state,next_a0,trans).reshape(nt, M2, tt, self.S_dims+1),self.sample_cond_trajectory(M2, tt, pi,next_state, next_a1,trans).reshape(nt, M2, tt, self.S_dims+1)])
        print('Done. cost {}'.format(time.time() - startT))
            
        state_star = trajs_star[:,:,:self.S_dims]# (n2, tt2)
        state_cond = trajs_cond[:,:,:,:self.S_dims]# (n1*tt1, m, tt2)
        state_next_cond = trajs_cond_next[:,:,:,:,:self.S_dims]# (dim_a, n1*tt1,m, tt2 )
        print('cal td error ...') 
        startT = time.time()
        # a = reward + self.gamma * v_func.get_v(next_state) - v_func.get_v(state)
        # print(a.shape)
        # b = v_func.get_a(state)#[:,action]
        # print(action.shape)
        # print(b.shape)
        td_error = reward + self.gamma * v_func.get_v(next_state) - v_func.get_v(state) - v_func.get_a(state)[range(self.nt),action] #(n1*tt1)
        print('v func value is ')
        print(v_func.get_v(state)[:20])
        print('Done. cost {}'.format(time.time() - startT))
        sstar = state_star.reshape(-1,self.S_dims)
        a1 = np.ones(sstar.shape[0])
        a0 = np.zeros(sstar.shape[0])
        print('cal cond omega 1 ...')
        startT = time.time()
        omega1 = omega.get_cond_omega(state,action,sstar,a1).reshape(state.shape[0],-1) #(n1*tt1, dim_base)
        print('Done. cost {}'.format(time.time() - startT))
        print('cal cond omega 0 ...')
        startT = time.time()
        omega0 = omega.get_cond_omega(state,action,sstar,a0).reshape(state.shape[0],-1) #(n1*tt1, dim_base)
        print('omega0 value is ')
        print(omega0[:10,:10])
        print('Done. cost {}'.format(time.time() - startT))
        otd1 = np.mean(omega1 * td_error[:,None],axis=0)#(n2*tt2)
        otd0 = np.mean(omega0 * td_error[:,None],axis=0)#(n2*tt2)
        star_td = np.stack([otd0.reshape(M,tt),otd1.reshape(M,tt)]).transpose(1,2,0)#(n2,tt2,dim_A)


        self.components = {
            'feature_dim':self.feature_dim,
            'state': state,            #(tn,)
            'state_star':state_star,   # (M, tt2)
            'state_cond':state_cond,   # ( n1*tt1,M,tt2)
            'state_next_cond':state_next_cond,  # (dim_a, n1*tt1,M, tt2,)

            'gammas':gammas,          # (1, tt2, 1 )
            'v_func':v_func,          # (dim_s,dim_a)
            'pi':pi,          # (dim_s,dim_a)
            'prob_next_state':pi.get_prob(next_state),  # (n1*tt1, dim_a)
            'omega':omega.get_omega(state, action),           # (n1*tt1, )

            'star_td': star_td, #(n2,tt2,dim_A)
            # 'td_error': td_error,                   #(n1*tt1)
        }


    # sample_trajectory(M, T_star, up_pm_old,trans)

    def sample_trajectory(self,num_trajs,tt, pi, trans=None):
        trajs = np.zeros((num_trajs, tt, self.S_dims+1))
        s = self.init_sampler(num_trajs)
        if trans == None :
            tr = self
        else:
            tr = trans
#         print('sample_trajs using trans {}'.format(trans))
        for t in range(tt):
            a = pi.sample_A(s)
            trajs[:,t] = np.concatenate([s,a.reshape(-1,1)],axis=1)
            s = tr.step(s,a)
        return trajs
        # sample_cond_trajectory(M2, T_star,up_pm_old,state,action,trans) 


    def sample_cond_trajectory(self, M ,tt, pi, state, action, trans=None):

        tn = state.shape[0]
        trajs = np.zeros((tn * M, tt, self.S_dims+1))
        s = np.repeat(state,M,axis=0)
        a = np.repeat(action.reshape(-1),M)
        trajs[:, 0] = np.concatenate([s,a.reshape(-1,1)],axis=1)
        if trans == None :
            tr = self
        else:
            tr = trans
#         print('sample_trajs using trans {}'.format(trans))
        for t in range(1,tt):
            s = tr.step(s, a)
            a = pi.sample_A(s)
            trajs[:,t] = np.concatenate([s,a.reshape(-1,1)],axis=1)
        
        # trajs = trajs.reshape(tn, M, tt, 2)
        return trajs
 
    def eval_trajs(self, trajs):
        num_trajs = trajs.shape[0]
        tt = trajs.shape[1]
#         r = np.zeros(num_trajs)
        
        r = self.reward(trajs[:,:,:self.S_dims], trajs[:,:,-1])
        gammas = np.array([self.gamma ** t for t in range(r.shape[1])]).reshape(1, -1)
        r = np.sum(r*gammas, axis=1)
#         for t in range(tt):
#             # r += self.reward(trajs[:,t,0], trajs[:,t,1])
#             r += self.reward(trajs[:,t,0], trajs[:,t,1]) * self.gamma**t

        return r

    def eval_policy(self, num_trajs, tt, pi, type = 'old'):
#         print('eval_policy method{}'.format(type))
        if type == 'old':
            trajs = self.sample_trajectory(num_trajs, tt, pi,None)
        if type == 'new':
            trajs = self.sample_trajectory(num_trajs, tt,None)
        return self.eval_trajs(trajs)


