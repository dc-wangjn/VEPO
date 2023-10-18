import pandas as pd
import numpy as np
import rpy2

from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter


class train_vl():
    def __init__(self, trajs, gamma,num_A, seed):
        self.trajs = trajs
        self.gamma = gamma
        self.S_dims = trajs[0][0][0].shape[0]
        self.num_A = num_A
        
        self.seed = seed
        
    def get_pdframe(self):
        temp = [[i+1,j+1]+y[0].tolist()+y[1:3]+[0.5] for i,x in enumerate(self.trajs) for j,y in enumerate(x) ]
            
        df = pd.DataFrame(temp, columns=['id','time']+['s{}'.format(i+1) for i in range(self.S_dims)]+['act','util','prob'])
        return df
    
    
    def get_rtrajs(self):
        df = self.get_pdframe()
#         print(df.head(50))
        with localconverter(robjects.default_converter + pandas2ri.converter):
            r_from_pd_df = robjects.conversion.py2rpy(df)
        return r_from_pd_df
    
    def train(self):
        infinite = importr('infiniteHorizon')
        
        rtrajs = self.get_rtrajs()
        state_value = robjects.r('''
           state_value=list(vars=paste('s',sep='',1:{}),num_lags=rep(1,{}),basis="gauss")
        '''.format(self.S_dims,self.S_dims))
        policy_class = robjects.r('''
           policy_class=list(vars=paste('s',sep='',1:{}),num_lags=rep(1,{}),basis="linear")
        '''.format(self.S_dims,self.S_dims))
        
        res = infinite.vl(data=rtrajs,id='id',util='util',act='act',
                          state_value=state_value,policy_class=policy_class,
                         disc=self.gamma,lambdas=1,K=1,center_scale = False,seed=self.seed)
        beta_ = np.array(res[0])
        beta = beta_.reshape(self.num_A-1,self.S_dims+1)

        return policy_vl(beta)

class policy_vl():
    def __init__(self, beta):
        self.beta = beta
        self.num_A = self.beta.shape[0]

    def get_A_prob(self,states,actions = None, multi_dim = False, to_numpy= True):
        n = states.shape[0]
        _states = np.concatenate([np.ones(n)[:,None],states],axis=1)
        logit = np.exp(np.matmul(_states,np.transpose(self.beta)))
        den = 1 + np.sum(logit,axis=1,keepdims=True)
        prob_last = 1/den
        prob = np.concatenate([logit/den, prob_last],axis=1)

        return prob
    
    def get_A(self, states):
        return self.sample_A(states)
    
    
    def sample_A(self, states):
        A_prob = self.get_A_prob(states) # , to_numpy = to_numpy
        sample_As = (A_prob.cumsum(1) > np.random.rand(A_prob.shape[0])[:,None]).argmax(1)   
        return sample_As
    
    
    
    
    