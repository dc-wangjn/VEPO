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


gamma = 0.95
feature_dim = 50

nt = (30,30)

M = 100
M2 = 10
T_star = 50
delta = 0.05
mod = 'origin'
ori_trans = True

cal_nu_nn = 50
cal_nu_tt = 50
cal_nu_mm = 10
# num_trajs = 200
# tt =50
p1 = np.ones(feature_dim)*0.1
p2 = np.ones(feature_dim)*0.5

p3 = np.ones(feature_dim)*0.7
P_old = [p1,p2,p3]
fqi_iters = 100

K_iters = 2

INIT_LOGLAMBDA = 3.9
INIT_LOGLAMBDA2 = 4.6

pi_old = P_old[0]

toy = VL_SIMU(gamma, feature_dim,delta)



pm_old = toy.learn_init_policy(nt[0],nt[1],fqi_iters, 'FQI')
value_old = np.mean(toy.eval_policy(10000, 100, pm_old, type = 'old'))
print('init value is {} '.format(value_old))
if ori_trans :
    trans = None
else:    
    trans = toy.cal_transition(nt[0],nt[1])
start_T = time.time()

v_func = toy.cal_q_function(pm_old,nt[0],nt[1])
omega = toy.cal_density_ratio(nt[0],nt[1], pm_old)

print('nuisance time cost {}'.format(time.time()-start_T))

pib = Random_Policy()
value_old = np.mean(toy.eval_policy(10000, 100, pib, type = 'old'))
print('behav value is {} '.format(value_old))
trajs_test = toy.sample_trajectory(nt[0],nt[1],pib,None)
start_T = time.time()
toy.compute_components(pm_old, cal_nu_nn,cal_nu_mm,cal_nu_tt,M,M2, T_star, trajs_behav = trajs_test, v_func = v_func ,omega = omega,  trans = trans)
print('components time cost {}'.format(time.time()-start_T))
#     toy.construct_policy(simple,w,h_dims = h_dims, lr_lambda = 1e-4
#                      , Adam_para = {"lr" : 5e-4, "w_clipping_val" : 0.5, "w_clipping_norm" : 1.0} )
parms = np.ones(feature_dim)*0.5
#     print(get_A_prob(parms,[0,1]))
    # print(parms[0])
lamb = 50
args = {'toy':toy,'lamb':50}


start_T = time.time()

result = minimize(compute_loss, parms,args, method='nelder-mead',options={'xatol': 1e-7, 'disp': True})
print('optimization time cost {}'.format(time.time()-start_T))
parms = result.x
print(parms)
improved_pi = Policy(parms,feature_dim)
value = np.mean(toy.eval_policy(10000, 100, improved_pi, type = 'old'))
print('improved value is {} '.format(value))

print('ITERATION 2:')
pm_old = Policy(parms,feature_dim)
value_old = np.mean(toy.eval_policy(10000, 100, pm_old, type = 'old'))
print('old value is {} '.format(value_old))

if ori_trans :
    trans = None
else:    
    trans = toy.cal_transition(nt[0],nt[1])
start_T = time.time()

v_func = toy.cal_q_function(pm_old,nt[0],nt[1])
omega = toy.cal_density_ratio(nt[0],nt[1], pm_old)

print('nuisance time cost {}'.format(time.time()-start_T))

pib = Random_Policy()
value_old = np.mean(toy.eval_policy(10000, 100, pib, type = 'old'))
print('behav value is {} '.format(value_old))
# trajs_test = toy.sample_trajectory(nt[0],nt[1],pib)
start_T = time.time()
toy.compute_components(pm_old, cal_nu_nn,cal_nu_mm,cal_nu_tt,M,M2, T_star, trajs_behav = trajs_test, v_func = v_func ,omega = omega,  trans = trans)
print('components time cost {}'.format(time.time()-start_T))
#     toy.construct_policy(simple,w,h_dims = h_dims, lr_lambda = 1e-4
#                      , Adam_para = {"lr" : 5e-4, "w_clipping_val" : 0.5, "w_clipping_norm" : 1.0} )
parms = np.ones(feature_dim)*0.5
#     print(get_A_prob(parms,[0,1]))
    # print(parms[0])
lamb = 50
args = {'toy':toy,'lamb':50}


start_T = time.time()

result = minimize(compute_loss, parms,args, method='nelder-mead',options={'xatol': 1e-7, 'disp': True})
parms = result.x
print(parms)
improved_pi = Policy(parms,feature_dim)
value = np.mean(toy.eval_policy(10000, 100, improved_pi, type = 'old'))
print('improved value is {} '.format(value))

print('ITERATION 3:')
pm_old = Policy(parms,feature_dim)
value_old = np.mean(toy.eval_policy(10000, 100, pm_old, type = 'old'))
print('old value is {} '.format(value_old))

if ori_trans :
    trans = None
else:    
    trans = toy.cal_transition(nt[0],nt[1])
start_T = time.time()

v_func = toy.cal_q_function(pm_old,nt[0],nt[1])
omega = toy.cal_density_ratio(nt[0],nt[1], pm_old)

print('nuisance time cost {}'.format(time.time()-start_T))

pib = Random_Policy()
value_old = np.mean(toy.eval_policy(10000, 100, pib, type = 'old'))
print('behav value is {} '.format(value_old))
# trajs_test = toy.sample_trajectory(nt[0],nt[1],pib)
start_T = time.time()
toy.compute_components(pm_old, cal_nu_nn,cal_nu_mm,cal_nu_tt,M,M2, T_star, trajs_behav = trajs_test, v_func = v_func ,omega = omega,  trans = trans)
print('components time cost {}'.format(time.time()-start_T))
#     toy.construct_policy(simple,w,h_dims = h_dims, lr_lambda = 1e-4
#                      , Adam_para = {"lr" : 5e-4, "w_clipping_val" : 0.5, "w_clipping_norm" : 1.0} )
parms = np.ones(feature_dim)*0.5
#     print(get_A_prob(parms,[0,1]))
    # print(parms[0])
lamb = 50
args = {'toy':toy,'lamb':50}


start_T = time.time()

result = minimize(compute_loss, parms,args, method='nelder-mead',options={'xatol': 1e-7, 'disp': True})
parms = result.x
print(parms)
improved_pi = Policy(parms,feature_dim)
value = np.mean(toy.eval_policy(10000, 100, improved_pi, type = 'old'))
print('improved value is {} '.format(value))