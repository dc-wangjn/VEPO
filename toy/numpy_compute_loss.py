import numpy as np
import os 
import time
import scipy


def construct_policy(parms):
    return np.array([[parms[0],1-parms[0]],[parms[1],1-parms[1]]])

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=-1,keepdims=True)
def sigmoid(x):
    return 1/(1+np.exp(-x))


def get_A_prob(parms, states, actions=None, multi_dim=True, to_numpy=True):
    
#     states = np.expand_dims(states,axis=-1)
#     dims = len(states.shape)
#     policy = construct_policy(parms)
#     print(states)
#     print(parms[0])
#     print(states*parms[0])
#     print(sigmoid(states*parms[0]+parms[1]))
    p_a1 = np.expand_dims(sigmoid(np.array(states)*parms[0]+parms[1]),axis=-1)
    p_a2 = 1 - p_a1
    prob = np.concatenate([p_a1,p_a2],axis=-1)
    if prob.any()>1:
        print('bug! prob is {}'.format(prob))
        print('sigmoid is {}'.format(sigmoid(np.array(states)*parms[0]+parms[1])))
        return 0
    return prob
    
    
#     w = np.expand_dims(parms[:2],axis=0)
#     b= parms[2:]
#     for i in range(dims-1):
#         b = np.expand_dims(b,axis=0)
        
#     print(states.shape)
#     print(parms[0].shape)
    probs = policy[states]
    return probs

def compute_loss(parms,args):
    
    self = args['toy']
#     real_idx = args['real_idx']
    lamb = args['lamb']

    state_star = self.components['state_star']
    a_func = self.components['a_func']
    gammas = self.components['gammas']
    pm_old = self.components['pm_old']
    
    


    #################### l1 ###########################

    a_star = a_func[state_star]#(n2, tt2, dim_A)
    pi_star = get_A_prob(parms, state_star, multi_dim=True,to_numpy=False)#(n2, tt2, dim_A)
    factor = np.sum(gammas)
    l1 = np.sum(a_star *pi_star*gammas, axis= (1,2))
    l1 = np.mean(l1)/factor

    #################### l2 ###########################
    omega_with_target = self.components['omega_with_target']#(batch, dim_base)

    td_error = self.components['td_error'] #(batch,)

    omega_td = np.mean(omega_with_target * td_error[:,None],axis=0) #(dim_base)

    s_0 = (state_star<<1)
    s_1 = (state_star<<1) + 1
    star_td = np.stack([omega_td[s_0],omega_td[s_1]]).transpose(1,2,0)
    
#     print(parms[0].shape)

    pi_star = get_A_prob(parms, state_star, multi_dim=True,to_numpy=False)  # (n2, tt2, dim_A)
    old_pi_star = pm_old[state_star]# (n2, tt2, dim_A)
    l2 = np.sum((pi_star - old_pi_star) * star_td * gammas,axis=(1,2))
    l2 = np.mean(l2)/factor/(1-self.gamma)

    #################### l3 ###########################

    state = self.components['state']
    state_next_cond = self.components['state_next_cond'][:,:] # (dim_a, n1*tt1,m, tt2 )
    state_cond = self.components['state_cond'] # (n1*tt1,m, tt2)
    next_pi = self.components['prob_next_state'] #(n1*tt1, dim_a)
    omega = self.components['omega']
    first_term = np.sum(get_A_prob(parms, state_next_cond, multi_dim=True,to_numpy=False) * a_func[state_next_cond] * gammas[None,None,],axis=(3,4)) #dim_a,batch
    first_term = np.mean(first_term, axis= 2)
    first_term = self.gamma * np.sum(np.transpose(first_term) * next_pi,axis = 1)/factor #( batch,)

    # (n1*tt1, m, tt2 ,dim_a)

    second_term = np.sum(get_A_prob(parms, state_cond, multi_dim=True,to_numpy=False) * a_func[state_cond] * gammas, axis=(2,3))
    second_term = np.mean(second_term,axis= 1) / factor # (batch,)

    #(n1*tt1, dim_a)
    third_term = np.sum(get_A_prob(parms, state, multi_dim=True,to_numpy=False) * a_func[state] ,axis=1) * (1 - self.gamma)
    l3 = np.mean((first_term - second_term + third_term) * omega)/(1-self.gamma)

    lift = l1 + l2 + l3
#     print('l1 is {}, l2 is {}, l3 is{}'.format(l1,l2,l3))
#     print(l2)
#     print(l3)
    def kl(old,new):
        return old * np.log(np.clip(old, 1e-6, 1)/ np.clip(new, 1e-6, 1))

    kl_divergence = kl(old_pi_star,pi_star)
    kl_loss = np.sum(kl_divergence * gammas, axis = (1,2))
    kl_loss = np.mean(kl_loss)/factor

    loss = -lift + np.maximum(kl_loss - self.KL_delta,0) * lamb
    return loss