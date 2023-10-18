import numpy as np
import time
from multiprocessing import Pool
from livelossplot import PlotLosses
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import time
import copy
import os
from scipy.optimize import minimize
from scipy.optimize import Bounds

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


    
from simulation_trpo import VL_SIMU
from numpy_compute_loss import compute_loss
from numpy_compute_loss import get_A_prob
from numpy_compute_loss import construct_policy
from modules import Policy
from modules import Random_Policy

import argparse

def init():
    global tf
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
def run_one_rep(rep, args):
    np.random.seed(rep)
    mod = args['mod']
    pi_old = args['pm_old']
    nt = args['nt']
    toy = args['toy']
    cal_nu_nn = args['cal_nu_nn']
    cal_nu_mm = args['cal_nu_mm']
    cal_nu_tt = args['cal_nu_tt']
    M = args['M']
    M2 = args['M2']
    T_star = args['T_star']

    v_func = args['v_func']
    omega = args['omega']
    trans = args['trans']
    w = args['w']
    h_dims = args['h_dims']
    batch = args['batch']
    multi_step = args['multi_step']
    improve_step = args['improve_step']
    TRUE_TRANS = args['TRUE_TRANS']
    K_iters = args['K_iters']
    noise_var = args['noise_var']
    eps = args['eps']
    simple = args['simple']
    feature_dim = args['feature_dim']
    
    res =  []
    ######
    print('using N_T:{}'.format(nt))
    pib = Random_Policy()
    trajs_test = toy.sample_trajectory(nt[0],nt[1],pib)
    toy.compute_components(pi_old, cal_nu_nn,cal_nu_mm,cal_nu_tt,M,M2, T_star, trajs_behav = trajs_test, v_func = v_func ,omega = omega,  trans = trans)
#     print('components time cost {}'.format(time.time()-start_T))
#     toy.construct_policy(simple,w,h_dims = h_dims, lr_lambda = 1e-4
#                      , Adam_para = {"lr" : 5e-4, "w_clipping_val" : 0.5, "w_clipping_norm" : 1.0} )
    parms = np.ones(feature_dim)*0.5
#     print(get_A_prob(parms,[0,1]))
    # print(parms[0])
    lamb = 50
    args = {'toy':toy,'lamb':50}

    # bounds = Bounds(lb=-10,ub=10)
    start_T = time.time()
    print('using_batch:{}'.format(batch))
    result = minimize(compute_loss, parms,args, method='nelder-mead',options={'xatol': 1e-8, 'disp': True})
    parms = result.x
#     print(parms)
    pm_improved = Policy(parms,feature_dim)
#     toy.improve_policy(simple = simple, multi_step = multi_step, max_iter = improve_step, batch_size = batch, print_freq = int(improve_step/10),eps = eps)
    print('optimization time cost {}'.format(time.time()-start_T))

    value_old = np.mean(toy.eval_policy(10000, 100, pi_old, type = 'old'))
    value_improved = np.mean(toy.eval_policy(10000, 100, pm_improved, type = 'old'))
    print('rep {},model is {},old value is {}, improved  value is {} '.format(rep,mod, value_old, value_improved))

    res.append([np.array([0,0]),np.array([value_old,value_improved]),0])

    for n7,t in enumerate(range(K_iters)):

#         s = np.array([0, 1])
        up_pm_old = Policy(parms,feature_dim)
        print('last iter pi {}'.format(up_pm_old))
        start_T = time.time()
        up_q = toy.cal_q_function(up_pm_old,nt[0],nt[1])
        # #                     print(q)
        up_omega = toy.cal_density_ratio(nt[0],nt[1], up_pm_old)
        trans = toy.cal_transition(nt[0],nt[1])
      
            
        print('mod {} ,transition is {}'.format(mod,trans))
        print('nuisance time cost {}'.format(time.time()-start_T))
        #                     print(omega_cond)


        start_T = time.time()
        toy.compute_components(up_pm_old, cal_nu_nn,cal_nu_mm,cal_nu_tt,M,M2, T_star, trajs_behav = trajs_test,
                              v_func = up_q ,omega = up_omega, trans = trans)
        print('components time cost {}'.format(time.time()-start_T))

        start_T =time.time()
        print('using_batch:{}'.format(batch))



        parms = np.ones(feature_dim)*0.5
#         print(get_A_prob(parms,[0,1]))
        # print(parms[0])
        lamb = 50
        args = {'toy':toy,'lamb':50}
        # bounds = Bounds(lb=-10,ub=10)
        start_T = time.time()
        print('using_batch:{}'.format(batch))
        result = minimize(compute_loss, parms,args, method='nelder-mead',options={'xatol': 1e-8, 'disp': True})
        parms = result.x
#         print(parms)
        pm_improved = Policy(parms,feature_dim)
    #     toy.improve_policy(simple = simple, multi_step = multi_step, max_iter = improve_step, batch_size = batch, print_freq = int(improve_step/10),eps = eps)
        print('optimization time cost {}'.format(time.time()-start_T))

        value_old = np.mean(toy.eval_policy(10000, 100, up_pm_old, type = 'old'))
        value_improved = np.mean(toy.eval_policy(10000, 100, pm_improved, type = 'old'))
        print('rep {},model is {},old value is {}, improved  value is {} '.format(rep,mod, value_old, value_improved))
        
        print('optimization time cost {}'.format(time.time()-start_T))
        # import matplotlib.pyplot as plt

        # pm_old = np.array([[1,0],[0,1]])
#         value_old = np.mean(toy.eval_policy(10000, 100, up_pm_old, type = 'old'))
#         value_improved = np.mean(toy.eval_policy(10000, 100, None, type = 'new'))
#         print('iter :{},rep {},model is {},old value is {}, improved  value is {} '.format(t,rep, mod, value_old, value_improved))
#         #                 print(toy.new_pi.get_A_prob(state))
#         #                 print(toy.new_pi.sample_A(state))
        res.append([np.array([0,0]),np.array([value_old,value_improved]),0])
    return res
def main(args):
    
    GAMMA = [0.9,0.95]
    r_sigma = args.sigma
    delta = args.delta
    delta2 = args.delta2
    feature_dim = args.feature_dim

    output_pth = 'results/est_multi_sigma{}_delta{}_{}.pkl'.format(r_sigma,args.delta,args.suffix)
#     NN_TT = [[10,10],[50,50]]
#     NN_TT = ((10,10),(30,30))
    # NN_TT = ((30,30),(50,50))
    NN_TT = [(30,30)]
#     NN_TT = ((50,50),(70,70))
#     NN_TT = ((10,10),(20,20))
#     NN_TT = [(10,10)]
    # NN = [10,10]
    # TT = [20,20]

    M = 100
    M2 = 10
    T_star = 50
    # model = ['origin','mod1','mod2','mod3','mod4']
    mod = 'origin'
#     model = ['mod4']
    cal_nu_nn = 100
    cal_nu_tt = 100
    cal_nu_mm = 10
    # num_trajs = 200
    # tt =50
    p1 = np.ones(feature_dim)*0.1
    p2 = np.ones(feature_dim)*0.5
#     p3 = np.array([[0,1],[1,0]])
    p3 = np.ones(feature_dim)*0.7
    P_old = [p1,p2,p3]
#     P_old = [p2]

    rep = args.rep
    K_iters = 2
#3,4.6,5.7
#     INIT_LOGLAMBDA = 4.6   #99
# #     INIT_LOGLAMBDA = 5.7
#     INIT_LOGLAMBDA2 = 5.7 #298
    INIT_LOGLAMBDA = 3.9
    INIT_LOGLAMBDA2 = 4.6

    multi_step = 1
    improve_step = 15000
    eps = 0
    batch = args.batch
    batch_change = args.batch_change
    
    noise_var = 2
    
    
    simple = True
    h_dims = 64
    
    config = {'feature_dim':feature_dim,'r_sigma':r_sigma,'gamma':GAMMA,'delta':delta,'delta2':delta2,'NN_TT':NN_TT,'M':M,'M2':M2,'T_star':T_star,'policy':P_old,'out_dir':output_pth,'rep':rep,'multi_step':multi_step,'improve_step':improve_step,'INIT_LOGLAMBDA':INIT_LOGLAMBDA,'INIT_LOGLAMBDA2':INIT_LOGLAMBDA2,'batch':batch,'batch_change':batch_change,'h_dims':h_dims,'noise_var':noise_var,'eps':eps,'simple':simple}
    print(config)
    result = np.zeros((len(GAMMA),K_iters+1,len(P_old),len(NN_TT),rep,2))
    lift = np.zeros((len(GAMMA),K_iters+1,len(P_old),len(NN_TT),rep,2))
    cr = np.zeros((len(GAMMA),K_iters+1,len(P_old),len(NN_TT),rep))


    if simple == True:
        w = [np.random.normal(0,0.5,(1,2))]
    else:
        
        w1 = np.random.normal(0,0.5,(1,h_dims))
        w2 = np.random.normal(0,1/h_dims/2,(h_dims,2))
        w = [w1,w2]
    for n1, gamma in enumerate(GAMMA) :


        for n2,pi_old in enumerate(P_old):
            if np.sum(pi_old==p3)==4:
#             if pm_old.any() == p3.any():
                print('using 2 setting!')
                delta = delta2
                INIT_LOGLAMBDA = INIT_LOGLAMBDA2
            toy = VL_SIMU(gamma, feature_dim, delta)


            for n5,nt in enumerate(NN_TT):
                pm_old = Policy(pi_old,feature_dim)
                TRUE_TRANS = toy.cal_transition(nt[0],nt[1])
                start_T = time.time()
            #     a_func,v_func = toy.cal_A_function(pm_old,cal_nu_nn,cal_nu_mm,cal_nu_tt)
                #                 print(a_func,v_func)
                q = toy.cal_q_function(pm_old,nt[0],nt[1])
                OMEGA = toy.cal_density_ratio(nt[0],nt[1], pm_old)

                print('nuisance time cost {}'.format(time.time()-start_T))


                omega= OMEGA
                v_func = q
                trans = TRUE_TRANS

                if batch_change == True:
                    batch = int(nt[0]*nt[1]/2)
#                 for n4,tt in enumerate(TT):
#                         for n6 in range(rep):
                args = {'feature_dim':feature_dim,'mod':mod,'pm_old':pm_old,'nt':nt,'toy':toy,'cal_nu_nn':cal_nu_nn,'cal_nu_mm':cal_nu_mm,'cal_nu_tt':cal_nu_tt,'M':M,'M2':M2,'T_star':T_star,'v_func':v_func,'omega':omega,'trans':trans,'w':w,'h_dims':h_dims,'batch':batch,'multi_step':multi_step,'improve_step':improve_step,'TRUE_TRANS':TRUE_TRANS,'K_iters':K_iters,'noise_var':noise_var,'eps':eps,'simple':simple}

                values = [[r,args] for r in range(rep)]
                with Pool(processes=3,initializer = init) as pool:
                    res = pool.starmap(run_one_rep,values)
                    
#                     print(res)
                for i in range(rep):
                    for j in range(K_iters+1):
                        lift[n1,j,n2,n5,i] = res[i][j][0]
                        result[n1,j,n2,n5,i] = res[i][j][1]
                        cr[n1,j,n2,n5,i] = res[i][j][2]
                            
                f = open(output_pth,'wb')
                pickle.dump([result,lift,cr,config],f)
                print('save result done!')
                f.close()
                    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_dim', type=int, default=10)
    parser.add_argument('--sigma', type=float, default=1)
    parser.add_argument('--delta', type=float, default=0.1)
    parser.add_argument('--delta2', type=float, default=0.1)
    parser.add_argument('--rep', type=int, default=5)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--batch_change', type=bool, default=False)
    parser.add_argument('--suffix', type=str)
    args = parser.parse_args()
    start_t = time.time()
    main(args)                   
    cost = time.time()-start_t
    print('total_time used {}'.format(cost))