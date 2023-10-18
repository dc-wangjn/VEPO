import pickle
import argparse


import ve_trpo_cv as TRPO
import simulation.Ohio_Simulator as Ohio
from _util import * 
import _RL.FQE as FQE_module
import _RL.FQIRF as  FQI_module
import _RL.true_Q as true_Q
reload(true_Q)

from _base_policy.BEAR import bear, utils
reload(Ohio)
reload(FQE_module)
reload(TRPO)
import tensorflow as tf
import copy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# https://trpo.notebook.us-east-1.sagemaker.aws/lab/proxy/6006/
def init():
    global tf
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
import os

##################generate random policy trajs to train optimal policy####################


def one_rep(gamma,trajs, rep_k, init_method,trans_method,use_ori_trans,delta,FQI_iter,BEAR_iter,improve_iter):
    T = 50
    N = 5000
    seed = rep_k * 10
   
    gamma = gamma
    L = 1
    sd_G = 3
    VE = TRPO.ValueEnhancement(normalize =True, L = L, gamma = gamma,seed = seed)#non_overlap_dim =3

    FQI_paras = {"max_iter" : 100, "eps" : 0 # FQI iter
            , "lr" : 1e-3, "decay_steps" : 100 # Adam
             , "hiddens" : [256, 256], "batch_size" : 16384, "max_epoch" : 20 # NN
             , "es_patience" : 1}

    rf_iter=100
    rf_eps=0.001
    max_depth=2
    warm_start=False
    time = now()
    VE.learn_optimal_policy(sd_Go=sd_G,T_o=T,N_o=N,verbose=1,eval_N=1000,test_freq=10,train_freq=10,
                            nn_verbose=0,validation_freq=5,latent_dim=32,train_iter=1,method='BEAR',
                            max_depth=max_depth,warm_start=True,
                            rf_iter=rf_iter,rf_eps=rf_eps,
                            **FQI_paras)
    print(now()-time)

    sd_G = 3


    VE.get_real_trajs(trajs)
    VE.get_behav(256)
    a = VE.eval_onpolicy(policy = "est_behav", simulator = "ohio")

    init_FQI_paras = {"max_iter" : FQI_iter, "eps" : 0.0001 # FQI iter
            , "lr" : 1e-3, "decay_steps" : 100 # Adam
             , "hiddens" : [256, 256], "batch_size" : 32, "max_epoch" : 1 # NN
             , "es_patience" : 1}
    VE.learn_init_policy(method=init_method, train_iter=BEAR_iter, **init_FQI_paras)
    dist =VE.eval_onpolicy(policy = "old", simulator = "ohio")


    FQE_paras = {"max_iter" : 1000, "eps" : 0.0005 # FQE iter
            , "lr" : 1e-3, "decay_steps" : 100 # Adam
             , "hiddens" : [256, 256], "batch_size" : 1024, "max_epoch" : 10 # NN
             , "gpu_number" : 0
             , "es_patience" : 1
             ,"stochastic" : True 
            , "update_plot": False}
    # eval the init policy with both simu and OPE
    VE.OPE(eval_N = 1000, test_freq = 10, verbose = 1, nn_verbose = 0, tag = 'old' , **FQE_paras)
    VE.est_density_ratio(h_dims = 256, max_iter=200, batch_size=32,eps=0.01, print_freq=5)
    VE.est_transition_density(v_dims = int(3), h_dims = int(64), epochs = 10000, batch_size = 64,trans_type=trans_method)
    """ TT here """
    VE.get_lower_bound_components(NN = 50, TT = 50, M = 5, G_ds_n_seeds = 1e3,use_ori_trans=use_ori_trans)

    VE.construct_obj_fun(lamb=5,h_dims = 128, lr_lambda = 1e-4
                         , Adam_para = {"lr" : 5e-5, "w_clipping_val" : 0.5, "w_clipping_norm" : 1,"beta":0.9}
                         , KL_delta = delta, depth=2)
    total_states = VE.components["total_states"]
    eval_state = total_states[:200]
    print('init model parms {}'.format(VE.policy.trainable_variables[0]))
    print('sample action under fix states {}'.format(VE.policy.sample_A(eval_state)))
    a = VE.eval_onpolicy(policy = "improved", simulator = "ohio")
    # max_iter = 1000 almost saturated...
    """better stop rule here"""
    VE.improve_policy(batch_size = 1024, max_iter =improve_iter, print_freq = int(improve_iter/5),sd_G = 3)
    print('__________________Iteration:{}_____________________'.format(VE.iter))
    a1 = VE.eval_onpolicy(policy = "improved", simulator = "ohio")
    b1 = VE.eval_onpolicy(policy = "old", simulator = "ohio")


    ##### second iteration ####
    
    VE.learn_init_policy(method = "last")
    VE.eval_onpolicy(policy = "old", simulator = "ohio")
    VE.OPE(eval_N = 1000, test_freq = 10, verbose = 1, nn_verbose = 0, tag = 'old' , **FQE_paras)


    VE.est_density_ratio(h_dims = 256, max_iter=200, batch_size=32,eps=0.01, print_freq=5)
    VE.est_transition_density(v_dims = int(3), h_dims = int(64), epochs = 10000, batch_size = 64,trans_type=trans_method)

    VE.get_lower_bound_components(NN = 50, TT = 50, M = 5, G_ds_n_seeds = 1e3,use_ori_trans=use_ori_trans)

    VE.construct_obj_fun(lamb=5,h_dims = 128, lr_lambda = 1e-4
                         , Adam_para = {"lr" : 5e-5, "w_clipping_val" : 0.5, "w_clipping_norm" : 1,"beta":0.9}
                         , KL_delta = delta, depth=2)
    total_states = VE.components["total_states"]
    eval_state = total_states[:200]
    print('init model parms {}'.format(VE.policy.trainable_variables[0]))
    print('sample action under fix states {}'.format(VE.policy.sample_A(eval_state)))
    a = VE.eval_onpolicy(policy = "improved", simulator = "ohio")
    # max_iter = 1000 almost saturated...
    """better stop rule here"""
    VE.improve_policy(batch_size = 1024, max_iter =improve_iter, print_freq = int(improve_iter/5),sd_G = 3)
    print('__________________Iteration:{}_____________________'.format(VE.iter))
    a2 = VE.eval_onpolicy(policy = "improved", simulator = "ohio")
    b = VE.eval_onpolicy(policy = "old", simulator = "ohio")
    
    ##### third iteration ####    
    VE.learn_init_policy(method = "last")
    VE.eval_onpolicy(policy = "old", simulator = "ohio")
    VE.OPE(eval_N = 1000, test_freq = 10, verbose = 1, nn_verbose = 0, tag = 'old' , **FQE_paras)


    VE.est_density_ratio(h_dims = 256, max_iter=200, batch_size=32,eps=0.01, print_freq=5)
    VE.est_transition_density(v_dims = int(3), h_dims = int(64), epochs = 10000, batch_size = 64,trans_type=trans_method)

    VE.get_lower_bound_components(NN = 50, TT = 50, M = 5, G_ds_n_seeds = 1e3,use_ori_trans=use_ori_trans)

    VE.construct_obj_fun(lamb=5,h_dims = 128, lr_lambda = 1e-4
                         , Adam_para = {"lr" : 5e-5, "w_clipping_val" : 0.5, "w_clipping_norm" : 1,"beta":0.9}
                         , KL_delta = delta, depth=2)
    total_states = VE.components["total_states"]
    eval_state = total_states[:200]

    print('init model parms {}'.format(VE.policy.trainable_variables[0]))
    print('sample action under fix states {}'.format(VE.policy.sample_A(eval_state)))
    a = VE.eval_onpolicy(policy = "improved", simulator = "ohio")
    # max_iter = 1000 almost saturated...
    """better stop rule here"""
    VE.improve_policy(batch_size = 1024, max_iter =improve_iter, print_freq = int(improve_iter/5),sd_G = 3)
    print('__________________Iteration:{}_____________________'.format(VE.iter))
    a3 = VE.eval_onpolicy(policy = "improved", simulator = "ohio")
    b = VE.eval_onpolicy(policy = "old", simulator = "ohio")
    final_action = VE.policy.sample_A(total_states).reshape(-1)
    
    
    final_dist = np.bincount(final_action)/len(final_action)    
    return np.array([np.mean(b1),np.mean(a1),np.mean(a2),np.mean(a3),VE.fail_counter]),final_dist

### some functions used to get trajs from source data ###

    
def Glucose2Reward( gls):
    low_gl, high_gl = 80, 140
    rewards = np.select([gls >= high_gl, gls <= low_gl, np.multiply(low_gl < gls, gls < high_gl)]
      , [-(gls - high_gl) ** 1.35 / 30, -(low_gl - gls) ** 2 / 30, 0])
    return rewards

def cal_reward( obs, gamma):
    all_rewards = Glucose2Reward(obs)
    all_rewards = all_rewards[0]
    all_rewards = np.roll(all_rewards[4:], shift = -1, axis = 0)
    gammas = np.expand_dims(arr([gamma ** j for j in range(all_rewards.shape[0])]), 0)
    discounted_values = np.dot(gammas, all_rewards)
    average_values = np.mean(all_rewards, 0)
    return discounted_values, average_values

def conc_SA_2_state(obs, actions, t, multiple_N = False, J = 4):
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


def make_traj(data):
    obs = data[:,1:-1].transpose()[:,:,None]
    actions = data[:,-1][:,None]
    trajs = []
    T = len(data)
    for t in range(4, T - 1): 
        conc_s = conc_SA_2_state(obs, actions, t, multiple_N = True, J = 4) # (dim_with_J, N)
        conc_ss = conc_SA_2_state(obs, actions, t + 1, multiple_N = True, J = 4)# (dim_with_J, N)
        As = actions[t, :]
        Rs = Glucose2Reward(conc_ss[-3, :])
        trajs.append([conc_s[:, 0], As[0], Rs[0], conc_ss[:, 0]])
    return trajs
###################################################################
def main(args):
    
    ### get trajs from source data
    path = os.getcwd() + "/code_data/code_data/Data_Ohio.csv"
    data = pd.read_csv(path, header = 0)
    data0 = np.array(data)
#     data1 = data0[0:1100,:]
#     data2 = data0[1100:2200,:]
#     data3 = data0[2200:3300,:]
#     data4 = data0[3300:4400,:]
#     data5 = data0[4400:5500,:]
#     data6 = data0[5500:6600,:]
    
    data1 = data0[0:700,:]
    data2 = data0[1100:1800,:]
    data3 = data0[2200:2900,:]
    data4 = data0[3300:4000,:]
    data5 = data0[4400:5100,:]
    data6 = data0[5500:6200,:]
    data_list = [data1,data2,data3,data4,data5,data6]
    real_trajs = [make_traj(x) for x in data_list]
    #####################################################

    rep = args.rep
    delta = args.delta
    epsilon = args.eps
    gamma = args.gamma
    
    
    #init policy settings
#     init_policy = ['FQI','CQL','behav']  # 'BEAR' ,'FQI', 'VL'
    init_policy = [args.init_policy]  # 'BEAR' ,'FQI', 'VL'
    FQI_iter = args.fqi_iter
    BEAR_iter = args.bear_iter
    
    improve_iter = args.improve_iter
    
    trans_method = args.trans_method  # 'GAN' or 'gauss'
    use_ori_trans = bool(args.ori_trans)   # True or False

    output_pth = 'results/realdata_gamma{}_delta{}_{}_{}_ori{}_{}.pkl'.format(gamma,delta,init_policy[0],trans_method,use_ori_trans,args.suffix)
    
    config = {'gamma':gamma,'init_policy':init_policy,'trans_method':trans_method,'use_ori_trans':use_ori_trans,'rep':rep,'delta':delta,'FQI_iter':FQI_iter,'BEAR_iter':BEAR_iter,'eps':epsilon}
    
    print(config)
    results = np.zeros((len(init_policy),rep,5))
    dists = np.zeros((len(init_policy),rep,5))
    for k,init_method in enumerate(init_policy):
        for r in range(rep):
            printG("<------ Start using  method {} with rep {} -------->\n".format(init_method,r))
            curr = now()
            results[k,r],dists[k,r] = one_rep(gamma,real_trajs, r, init_method,trans_method, use_ori_trans,delta,FQI_iter,BEAR_iter,improve_iter)
            printG("<------  method {} with rep {} DONE! Time cost = {:.1f} minutes -------->\n".format(init_method,r, (now() - curr) / 60))
            f = open(output_pth,'wb')
            pickle.dump([results,dists,config],f)
            f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--improve_iter', type=int, default=3000)
    parser.add_argument('--init_policy', type=str, default='FQI') # 'BEAR' ,'FQI', 'VL'
    parser.add_argument('--rep', type=int, default=100)
    parser.add_argument('--bear_iter', type=int, default=100)
    parser.add_argument('--fqi_iter', type=int, default=100)
    parser.add_argument('--trans_method', type=str, default='gauss') #' GAN' or 'gauss'
    parser.add_argument('--ori_trans', type=int, default=0) # 1 or 0
    parser.add_argument('--delta', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--eps', type=float, default=0.3)
    parser.add_argument('--suffix', type=str)
    args = parser.parse_args()
    start_t = time.time()
    main(args)                   
    cost = time.time()-start_t
    print('total_time used {}'.format(cost))


    
    