import numpy as np
import time
from livelossplot import PlotLosses
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import time

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
from toy import TOY

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sigma', type=float, default=1)
parser.add_argument('--rep', type=int, default=5)
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--batch_change', type=bool, default=False)
parser.add_argument('--suffix', type=str)
args = parser.parse_args()




GAMMA = [0.9]
r_sigma = args.sigma
delta = 0.1

output_pth = 'results/true_value_sigma{}_{}.pkl'.format(r_sigma,args.suffix)

# NN = [2,4,10,25,75]
# TT = [2,4,10,25,75]
NN_TT = ((10,10),(20,20))
# TT = [10,20]


M = 100
M2 = 10
T_star = 50

cal_nu_nn = 5000
cal_nu_tt = 100
cal_nu_mm = 10
# num_trajs = 200
# tt =50
p1 = np.array([[0.8,0.2],[0.2,0.8]])
p2 = np.array([[0.5,0.5],[0.5,0.5]])
p3 = np.array([[0,1],[1,0]])
P_old = [p1,p2,p3]

K_iters = [2]

rep = args.rep

INIT_LOGLAMBDA = 3

multi_step = 5
improve_step = 1000
batch = args.batch
batch_change = args.batch_change

h_dims = 64
config = {'r_sigma':r_sigma,'gamma':GAMMA,'delta':delta,'NN_TT':NN_TT,'M':M,'M2':M2,'T_star':T_star,'policy':P_old,'out_dir':output_pth,'rep':rep,'multi_step':multi_step,'improve_step':improve_step,'INIT_LOGLAMBDA':INIT_LOGLAMBDA,'batch':batch,'batch_change':batch_change,'h_dims':h_dims}
result = np.zeros((len(GAMMA),len(K_iters),K_iters[-1]+1,len(P_old),len(NN_TT),rep,2))
lift = np.zeros((len(GAMMA),len(K_iters),K_iters[-1]+1,len(P_old),len(NN_TT),rep,2))


w1 = np.random.normal(0,0.5,(1,h_dims))
w2 = np.random.normal(0,1/h_dims/2,(h_dims,2))
w = [w1,w2]

for n7, gamma in enumerate(GAMMA) :
    toy = TOY(gamma, delta, INIT_LOGLAMBDA,r_sigma)
    # pm_old = np.array([[0,1],[1,0]])
    # pm_old = np.array([[1,0],[0,1]])


    for n2,pm_old in enumerate(P_old):
        start_T = time.time()
        a_func,v_func = toy.cal_A_function(pm_old,cal_nu_nn,cal_nu_mm,cal_nu_tt)
        #                 print(a_func,v_func)
    #     q = toy.cal_q_function(pm_old,cal_nu_nn,cal_nu_mm,cal_nu_tt)
        #                 print(q)
        omega, omega_cond = toy.cal_density_ratio(cal_nu_nn,cal_nu_mm,cal_nu_tt, pm_old ,toy.pm)
        print('nuisance time cost {}'.format(time.time()-start_T))
        trajs_star = toy.sample_trajectory(M, T_star, pm_old)

        state = np.array([0,0,1,1])
        action = np.array([0,1,0,1])
        trajs_cond_temp = toy.sample_cond_trajectory(M2, T_star,pm_old,state,action).reshape(4, M2, T_star, 2)

        for n1,k in enumerate(K_iters):    
            for n3,nt in enumerate(NN_TT):
                if batch_change == True:
                    batch = int(nt[0]*nt[1]/2)
#                 for n4,tt in enumerate(TT):
                for n6 in range(rep):
                    print('using gamma {}'.format(gamma))
#                         print('using K_iter:{}'.format(k))
                    print('using init_pm:{}'.format(pm_old))
                    print('using N_T:{}'.format(nt))
#                         print('using tt:{}'.format(tt))
                    trajs_test = toy.sample_trajectory(nt[0],nt[1],toy.pm)

    #                 print('init_pm:{}'.format(pm_old))
                    start_T = time.time()

                    toy.compute_components(pm_old, cal_nu_nn,cal_nu_mm,cal_nu_tt,M,M2, T_star, trajs_behav = trajs_test,a_func= a_func, v_func = v_func ,omega = omega, omega_cond = omega_cond, trajs_star= trajs_star, trajs_cond_temp = trajs_cond_temp )
                    print('components time cost {}'.format(time.time()-start_T))
                    toy.construct_policy(w,h_dims = h_dims, lr_lambda = 1e-4
                                         , Adam_para = {"lr" : 5e-4, "w_clipping_val" : 0.5, "w_clipping_norm" : 1.0}
                                        )
                    start_T = time.time()
                    toy.improve_policy(multi_step = multi_step, max_iter = improve_step, batch_size = batch, print_freq = int(improve_step/3))
                    print('optimization time cost {}'.format(time.time()-start_T))

    #                 x = range(len(toy.losses))
    #                 plt.plot(x,np.array(toy.losses).reshape(-1))
    #                 plt.plot(x,toy.kl_losses)

                    # pm_old = np.array([[0.5,0.5],[0.25,0.75]])
                    # pm_old = np.array([[1,0],[0,1]])
                    value_old = np.mean(toy.eval_policy(10000, 100, pm_old, type = 'old'))
                    value_improved = np.mean(toy.eval_policy(10000, 100, None, type = 'new'))
                    print('rep {}, old value is {}, improved  value is {} '.format(n6,value_old, value_improved))

                    lift[n7,n1,0,n2,n3,n6] = np.array([toy.lift[-1],toy.kl_losses[-1]])
                    result[n7,n1,0,n2,n3,n6] = np.array([value_old,value_improved])
    #                 print(toy.new_pi.get_A_prob(state))
    #                 print(toy.new_pi.sample_A(state))
                    f = open(output_pth,'wb')
                    pickle.dump([result,lift,config],f)
                    print('save result done!')
                    f.close()
                    for n5,t in enumerate(range(k)):
#                         print('iter :{}'.format(t))
                        s = np.array([0, 1])
                        up_pm_old = toy.new_pi.get_A_prob(s)
                        print('last iter pi {}'.format(up_pm_old))
                        start_T = time.time()
                        up_a_func,up_v_func = toy.cal_A_function(up_pm_old,cal_nu_nn,cal_nu_mm,cal_nu_tt)
    #                     print(a_func,v_func)
    #                     q = toy.cal_q_function(up_pm_old,cal_nu_nn,cal_nu_mm,cal_nu_tt)
    # #                     print(q)
                        up_omega, up_omega_cond = toy.cal_density_ratio(cal_nu_nn,cal_nu_mm,cal_nu_tt, up_pm_old ,toy.pm)
                        print('nuisance time cost {}'.format(time.time()-start_T))
    #                     print(omega_cond)
                        up_trajs_star = toy.sample_trajectory(M, T_star, up_pm_old)

                        state = np.array([0,0,1,1])
                        action = np.array([0,1,0,1])
                        up_trajs_cond_temp = toy.sample_cond_trajectory(M2, T_star,up_pm_old,state,action).reshape(4, M2, T_star, 2)
                        start_T = time.time()
                        toy.compute_components(up_pm_old, cal_nu_nn,cal_nu_mm,cal_nu_tt,M,M2, T_star, trajs_behav = trajs_test,a_func= up_a_func, v_func = up_v_func ,omega = up_omega, omega_cond = up_omega_cond, trajs_star= up_trajs_star, trajs_cond_temp = up_trajs_cond_temp)
                        print('components time cost {}'.format(time.time()-start_T))
                        toy.construct_policy(w,h_dims = h_dims, lr_lambda = 1e-4
                                             , Adam_para = {"lr" : 5e-4, "w_clipping_val" : 0.5, "w_clipping_norm" : 1.0}
                                            )
                        start_T =time.time()
                        toy.improve_policy(multi_step = multi_step, max_iter = improve_step, batch_size = batch, print_freq = int(improve_step/3))
                        print('optimization time cost {}'.format(time.time()-start_T))
                        # import matplotlib.pyplot as plt

                        # pm_old = np.array([[1,0],[0,1]])
                        value_old = np.mean(toy.eval_policy(10000, 100, up_pm_old, type = 'old'))
                        value_improved = np.mean(toy.eval_policy(10000, 100, None, type = 'new'))
                        print('iter :{},rep {}, old value is {}, improved  value is {} '.format(t,n6, value_old, value_improved))
                        lift[n7,n1,n5+1,n2,n3,n6] = np.array([toy.lift[-1],toy.kl_losses[-1]])
                        result[n7,n1,n5+1,n2,n3,n6] = np.array([value_old,value_improved])
                        f = open(output_pth,'wb')
                        pickle.dump([result,lift,config],f)
                        print('save result done!')
                        f.close()