from _util import *

import simulation.Ohio_Simulator as Ohio
import _RL.sampler as sampler
import _RL.FQE as FQE_module
import _RL.FQI as FQI_module
import _RL.CQL as CQL_module
import _RL.FQIRF as FQIRF
import _RL.behavior_cloning as bc
from _base_policy.BEAR import bear_old, utils
from _base_policy.vlearning import vl_simu

import copy

reload(bear_old)

# from _base_policy.behavior_regularized_offline_rl.brac import brac_dual_agent
# from _base_policy.behavior_regularized_offline_rl.brac.utils import traj2dataset
# from _base_policy.behavior_regularized_offline_rl.brac.utils import shuffle_indices_with_steps
# from _base_policy.behavior_regularized_offline_rl.brac.utils import Flags

# from _density import GAN, omega_SASA_new, gaussian #new version
from _density import GAN, omega_SASA_new, gaussian #old version

reload(omega_SASA_new)
# reload(omega_SASA_latest)

reload(bc)
reload(Ohio)

import policy

reload(policy)
# importlib.reload(np)

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


############################################################################################################
############################################################################################################


class mixture_policy():
    def __init__(self, epsilon, opt_policy, base_prob):
        self.opt_policy = opt_policy
        self.epsilon = epsilon
        self.base_prob = base_prob
    
    def sample_A(self, states):
        mix_prob = self.get_A_prob(states)
        sample_As = (mix_prob.cumsum(1) > np.random.rand(mix_prob.shape[0])[:,None]).argmax(1)
        return sample_As
    
    def get_A(self, states):
#         mix_prob = self.get_A_prob(states)
        return self.sample_A(states)
#         return np.argmax(mix_prob,axis=-1)
    
#     def get_A(self, states):
#         if len(states.shape) == 1:
#             states = np.expand_dim(states, 1)
#         return np.argmax(self.model(states), axis = 1)
    
    def get_A_prob(self, states,actions = None, multi_dim = False, to_numpy= True):
        N = len(states)
        try:
            opt_prob = self.opt_policy.get_A_prob(states).cpu().numpy()
        except:
            opt_prob = self.opt_policy.get_A_prob(states)
          
        bs_prob = np.repeat(np.expand_dims(self.base_prob,axis=0),N,axis=0)
        mix_prob = opt_prob * self.epsilon + bs_prob * (1-self.epsilon)
        
        return mix_prob
        


class ValueEnhancement():
    """ Given a batch dataset (and the behaviour policy), improve the currect / (estimmated) initial policy
    1. improve the given initial (multi-step improvements)
    2. improve the estimated initial (get a good policy / multi-step improvements) (for our experiments too)
    policy_class # target(states, beta) = pi_beta(states); specify? or in what way?
    """

    def __init__(self,normalize, behav=None,non_overlap_dim=3,simulator='ohio',
                 mode_init_sampler='fit_emprical',L=3,gamma=0.9,seed=1,curr_policies=None):
        self.normalize = normalize
        self.non_overlap_dim = non_overlap_dim
        self.gamma = gamma
        self.L = L
        self.seed = seed
        self.mode_init_sampler = mode_init_sampler
        if curr_policies :
            self.curr_polies = curr_policies
        self.simulator = simulator
        self.fail_counter = 0
        
    
    def learn_optimal_policy(self, sd_Go, T_o, N_o, verbose=1, eval_N=1000,
                             test_freq=10, train_freq=10, 
                             nn_verbose=0,validation_freq=5,
                             latent_dim=32,train_iter=10,
                             method=None,min_q_weight=1.0,
                             max_depth=2,random_state=0,warm_start=True,
                             rf_iter=100,rf_eps=0.001,
                             **kwargs):


        curr_time = now()

#         if method == "BEAR":
        """ more dim and depth here"""
        ohio_rand = Ohio.OhioSimulator(sd_G = sd_Go, T = T_o, N = N_o)
        trajs_rand = ohio_rand.simu_one_seed(self.seed)
###################################################################################
        S_dims = len(trajs_rand[0][0][0])
        init_state = np.array([x[0][0] for x in trajs_rand])
        states, actions, rewards,next_states = [
            np.array([item[i] for traj in trajs_rand for item in traj]) for i in range(4)]
        self.s_max = np.max(states, axis = 0)
        self.s_min = np.min(states, axis = 0)
        if self.normalize == True:
            self.init_state = (init_state - self.s_min)/(self.s_max - self.s_min)
            trajs_rand = self.init_normalization(trajs_rand)
        else:
            self.init_state = init_state
        
        self.A_range = set(actions)
        self.num_A = len(self.A_range)
        if method == 'BEAR':

            """ more dim and depth here"""
            flatten_traj = sum(trajs_rand, [])
            replay_buffer = utils.ReplayBuffer(state_dim=S_dims, action_dim=1)
            for tup in flatten_traj:
                replay_buffer.add(tup)
            # why 5 will fail?
            self.optimal_policy = bear_old.BEAR(seed = self.seed,num_qs=2, latent_dim=latent_dim, state_dim=S_dims, action_dim=1, num_A=self.num_A, delta_conf=0.1 , use_bootstrap=False, version=0, lambda_=0.4, threshold=0.05, mode="auto" , num_samples_match=10, mmd_sigma=10.0,lagrange_thresh=10.0, use_kl=False, use_ensemble=True, kernel_type='laplacian')
            self.optimal_policy.train(replay_buffer, iterations=int(train_iter))  
        elif method =='FQI':

            optimal_pi_value_func = FQI_module.FQI(seed=self.seed,kernel=0,num_actions=self.num_A, gamma=self.gamma,init_states=self.init_state , **kwargs)
            opt_time = now()
            print('Start train optimal policy...')
            optimal_pi_value_func.train(trajs_rand, verbose=verbose, test_freq=test_freq,train_freq=train_freq, nn_verbose=nn_verbose, validation_freq=validation_freq)
            self.optimal_policy = optimal_pi_value_func
            print('train done!')
            disc_values = optimal_pi_value_func.init_state_value(init_states=self.init_state)
            ######


            printR("FQI : mean = {:.2f}".format(np.mean(disc_values)))    

            printG("<------------- optimal policy DONE! Time cost = {:.1f} minutes -------------> \n".format((now() - opt_time) / 60))
        elif method =='CQL':

            optimal_pi_value_func = CQL_module.CQL(kernel=0,min_q_weight=min_q_weight,seed=self.seed,num_actions=self.num_A, gamma=self.gamma,init_states=self.init_state , **kwargs)
            opt_time = now()
            print('Start train optimal policy...')
            optimal_pi_value_func.train(trajs_rand, verbose=verbose, test_freq=test_freq,train_freq=train_freq, nn_verbose=nn_verbose, validation_freq=validation_freq)
            self.optimal_policy = optimal_pi_value_func
            print('train done!')
            disc_values = optimal_pi_value_func.init_state_value(init_states=self.init_state)
            ######


            printR("CQL : mean = {:.2f}".format(np.mean(disc_values)))    

            printG("<------------- optimal policy DONE! Time cost = {:.1f} minutes -------------> \n".format((now() - opt_time) / 60))
            
        else:

            optimal_pi_value_func = FQIRF.FQI(
                num_actions=self.num_A,gamma=self.gamma,init_states=self.init_state,
                max_depth=max_depth,random_state=random_state,
                warm_start=warm_start,max_iter=rf_iter, eps=rf_eps)
            opt_time = now()
            print('Start train random forest optimal policy...')
            optimal_pi_value_func.train(trajs_rand, verbose=verbose, test_freq=test_freq,train_freq=train_freq)
            self.optimal_policy = optimal_pi_value_func
            print('train done!')
            disc_values = optimal_pi_value_func.init_state_value(init_states=self.init_state)
            ######


            printR("FQI : mean = {:.2f}".format(np.mean(disc_values)))    

            printG("<------------- optimal policy DONE! Time cost = {:.1f} minutes -------------> \n".format((now() - opt_time) / 60))            
    
        ##################################################
    ##################################################
    
    def init_normalization(self, trajs):
        
        N = len(trajs)
        T = len(trajs[0])
        states, actions, rewards, next_states = [
            np.array([item[i] for traj in trajs for item in traj]) for i in range(4)]  

        s_norm = (states - self.s_min)/(self.s_max - self.s_min)
        sn_norm = (next_states - self.s_min)/(self.s_max - self.s_min)
        trajs_norm = [[] for i in range(N)]
        for t in range(T):
            for i in range(N):
                trajs_norm[i].append([s_norm[t+i*T],actions[t+i*T],
                                      rewards[t+i*T],sn_norm[t+i*T]])
        return trajs_norm
    
    
    ### Parallel to method generate_behav, can not use in the same time 
    def get_real_trajs(self, trajs):
        self.trajs_ori = trajs
        self.N = len(trajs)
        self.T = len(trajs[0])
        self.ohio = Ohio.OhioSimulator(sd_G = 3, T = self.T, N = self.N, s_min = self.s_min,s_max=self.s_max)
        self.ohio_eval = Ohio.OhioSimulator(sd_G=3, T=100, N=10000, s_min = self.s_min,s_max=self.s_max)

        if self.normalize:
            print('now normalize')
            self.trajs = self.init_normalization(trajs)
        else:
            self.trajs = trajs        
    
        self.states, self.actions, self.rewards, self.next_states = [
            np.array([item[i] for traj in self.trajs for item in traj]) for i in range(4)]        
        self.S_dims = len(self.trajs[0][0][0])
        self.A_range = set(self.actions)
        self.A_dim = 1
        self.num_A = len(self.A_range)
        self.iter = 0
        

#         self.split_ind = sample_split(self.L, self.N)  # split_ind[k] = {"train_ind" : i, "test_ind" : j}

        self.split_ind = {}
        for k in range(self.L):
            self.split_ind[k] = {"train_ind" : np.arange(self.N), "test_ind" : np.arange(self.N)} 
        self.init_sampler = sampler.InitialStateSampler(data=[traj[0][0] for traj in self.trajs],
                                                        mode=self.mode_init_sampler)    
    
    
    
    def generate_behave(self,N,T,sd_G,epsilon):


        self.N, self.T = N,T 
        self.epsilon = epsilon
        self.ohio = Ohio.OhioSimulator(sd_G = sd_G, T = T, N = N, s_min = self.s_min,s_max=self.s_max)
        self.ohio_eval = Ohio.OhioSimulator(sd_G=sd_G, T=100, N=10000, s_min = self.s_min,s_max=self.s_max)
  
        trajs,_,_ = self.ohio.generate_mixture_policy_trajectory(self.optimal_policy, epsilon, gamma = self.gamma, seed = self.seed)
        self.trajs_ori = trajs
        
        if self.normalize:
            print('now normalize')
            self.trajs = self.init_normalization(trajs)
        else:
            self.trajs = trajs
            
        self.states, self.actions, self.rewards, self.next_states = [
            np.array([item[i] for traj in self.trajs for item in traj]) for i in range(4)]        
        self.S_dims = len(self.trajs[0][0][0])
        self.A_range = set(self.actions)
        self.A_dim = 1
        self.num_A = len(self.A_range)
        self.iter = 0
        

#         self.split_ind = sample_split(self.L, self.N)  # split_ind[k] = {"train_ind" : i, "test_ind" : j}
        self.split_ind = {}
        for k in range(self.L):
            self.split_ind[k] = {"train_ind" : np.arange(self.N), "test_ind" : np.arange(self.N)} 

        self.init_sampler = sampler.InitialStateSampler(data=[traj[0][0] for traj in self.trajs],
                                                        mode=self.mode_init_sampler)

    ################################################################################################

    def estimate_behav(self, states,actions, hidden_dims=32, depth=1):

        curr = now()
        behav = bc.BehaviorCloning(num_actions=self.num_A, hidden_dims=hidden_dims, S_dims=self.S_dims, depth=depth
                                        , activation='relu'
                                        , lr=5e-4, decay_steps=100000, batch_size=64, max_epoch=200
                                        , validation_split=0.2, patience=20, verbose=0)
        behav.train(states, actions)
        print("<------------- Behaviour Policy DONE! Time cost = {:.1f} ------------->".format((now() - curr) / 60))
        return behav
    
    def get_behav(self,  hidden_dims=32, depth=1):

        curr = now()
        self.behav = bc.BehaviorCloning(num_actions=self.num_A, hidden_dims=hidden_dims, S_dims=self.S_dims, depth=depth
                                        , activation='relu'
                                        , lr=5e-4, decay_steps=100000, batch_size=64, max_epoch=200
                                        , validation_split=0.2, patience=20, verbose=0)
        self.behav.train(self.states, self.actions)
        print("<------------- Behaviour Policy DONE! Time cost = {:.1f} ------------->".format((now() - curr) / 60))
#         return behav

    def learn_init_policy(self, method=None, min_q_weight=1.0,train_iter=1e2, do_CV=False, latent_dim=32,
                          verbose=1, eval_N=1000,test_freq=10, train_freq=10, 
                             nn_verbose=0,validation_freq=5,
                          max_depth=2,random_state=0,warm_start=False,**kwargs):
        """ step 2: Apply the existing state-of-the- art policy search methods to obtain an initial estimator [pre-fixed policy family?]
        """
        curr_time = now()
        self.init_policy = method
        self.curr_policy = []
        # for k in range(self.L):
        #     train_ind = self.split_ind[k]["train_ind"]
        if method == "BEAR":
            """ more dim and depth here"""
            
            
            flatten_traj = sum(self.trajs, [])

            replay_buffer = utils.ReplayBuffer(state_dim=self.S_dims, action_dim=1)
            for tup in flatten_traj:
                replay_buffer.add(tup)
            # why 5 will fail?
            curr_policy = bear_old.BEAR(seed = self.seed,num_qs=2,latent_dim=latent_dim,
                                            state_dim=self.S_dims, action_dim=1, 
                                            num_A=self.num_A, delta_conf=0.1, 
                                            use_bootstrap=False, version=0, 
                                            lambda_=0.4, threshold=0.05, 
                                            mode="auto", num_samples_match=10, 
                                            mmd_sigma=10.0,lagrange_thresh=10.0, 
                                            use_kl=False, use_ensemble=True, 
                                            kernel_type='laplacian')
            curr_policy.train(replay_buffer, iterations=int(train_iter))  # pol_vals =
        elif method == 'FQI':
            curr_policy = FQI_module.FQI(seed=self.seed,kernel=0,num_actions=self.num_A, gamma=self.gamma,init_states=self.init_state , **kwargs)
            opt_time = now()
            print('Start train init policy...')
            curr_policy.train(self.trajs, verbose=verbose, test_freq=test_freq,train_freq=train_freq, nn_verbose=nn_verbose, validation_freq=validation_freq)

            print('train done!')
        elif method == "CQL":

            curr_policy = CQL_module.CQL(kernel=0,min_q_weight=min_q_weight,seed=self.seed,num_actions=self.num_A, gamma=self.gamma,init_states=self.init_state , **kwargs)
            opt_time = now()
            print('Start train init policy...')
            curr_policy.train(self.trajs, verbose=verbose, test_freq=test_freq,train_freq=train_freq, nn_verbose=nn_verbose, validation_freq=validation_freq)

            print('train done!')
        
        #### VL may encounter the sigular matrix during optimization which can not be handled in our algorithm
        #### in order to keep the algorithm running, we have to use an alternative one.(CQL)
        
        elif method == 'VL':
        
            try:
                vlearn = vl_simu.train_vl(trajs = self.trajs, gamma = self.gamma)

                print('Start train vl policy...')
                curr_time = now()
                curr_policy = vlearn.train()
            except:
                print('VL failed, have to use CQL...')
                self.fail_counter +=1
                curr_policy = CQL_module.CQL(kernel=0,min_q_weight=min_q_weight,seed=self.seed,num_actions=self.num_A, gamma=self.gamma,init_states=self.init_state , **kwargs)
                opt_time = now()
                print('Start train init policy...')
                curr_policy.train([self.trajs[i] for i in train_ind], verbose=verbose, test_freq=test_freq,train_freq=train_freq, nn_verbose=nn_verbose, validation_freq=validation_freq)

                print('train done!')

        elif method == 'MIX':
            curr_policy = mixture_policy(self.epsilon, self.optimal_policy, self.ohio.p_A)

        elif method == 'RF':

            curr_policy = FQIRF.FQI(
                num_actions=self.num_A,gamma=self.gamma,init_states=self.init_state,
                max_depth=max_depth,random_state=random_state,
                warm_start=warm_start,max_iter=train_iter, eps=0.0001)
            opt_time = now()
            print('Start train init rf policy...')
            curr_policy.train(self.trajs, verbose=verbose, test_freq=test_freq,train_freq=train_freq)

            print('train done!')

        elif method == "behav":

            states, actions, rewards, next_states = [
                np.array([item[i] for traj in self.trajs for item in traj]) for i in range(4)]
            curr_policy = self.estimate_behav(states,actions,256)
        elif method == "last":
            curr_policy = self.policy
        for k in range(self.L):

            self.curr_policy.append(curr_policy)

        print("<------------- Initial Policy method {} DONE! Time cost = {:.1f} ------------->".format(method, (now() - curr_time) / 60))


    
    def OPE(self, verbose=1, eval_N=1000
            , test_freq=10, train_freq=10, nn_verbose=0, validation_freq=5
            ,tag = 'new', **kwargs):
        """ step 3: `FQE`
        TODO:
            1. need records for policy comparison?
        """
        self.curr_pi_value_funcs = []
        self.eval_N = eval_N
        disc_value_Ls = []
        for k in range(self.L):
            """ more dim and depth here"""
            curr_time = now()
            train_ind = self.split_ind[k]["train_ind"]
#             if tag == 'new':
# #                 print('FQE use new model')
# #                 curr_pi_value_func = FQE_module_new.FQE(policy=self.curr_policy, num_actions=self.num_A, gamma=self.gamma,init_states=self.init_state ,nn_verbose = nn_verbose, **kwargs)
# #                 curr_pi_value_func.train([self.trajs[i] for i in train_ind], verbose=verbose, test_freq=test_freq)  

#                 curr_pi_value_func = FQE_wjn.FQE(seed=self.seed,policy=self.curr_policy[k],num_actions=self.num_A, gamma=self.gamma,init_states=self.init_state , **kwargs)
#                 opt_time = now()
#                 print('Start train FQE...')
#                 curr_pi_value_func.train([self.trajs[i] for i in train_ind], verbose=verbose, test_freq=test_freq,train_freq=train_freq, nn_verbose=nn_verbose, validation_freq=validation_freq)
#                 print('train done!')


            # else:
                
            curr_pi_value_func = FQE_module.FQE(seed=self.seed,kernel=0,policy=self.curr_policy[k],  num_actions=self.num_A, gamma=self.gamma,init_states=self.init_state , **kwargs)
            curr_pi_value_func.train([self.trajs[i] for i in train_ind], verbose=verbose, test_freq=test_freq,train_freq=train_freq, nn_verbose=nn_verbose, validation_freq=validation_freq)
            self.curr_pi_value_funcs.append(curr_pi_value_func)
            # actually it is not fair to use ohio_eval.init_state here, becuse it does not take the errors in G into consideration
            disc_values = curr_pi_value_func.init_state_value(init_states=self.init_state)
            ######
            disc_value_Ls.append(np.mean(disc_values))

            printR("OPE : mean = {:.2f} and std = {:.2f}".format(np.mean(disc_values),
                                                                 np.std(disc_values) / np.sqrt(self.eval_N)))
            printG("<------------- initial OPE for fold {} DONE! Time cost = {:.1f} minutes -------------> \n".format(k,(now() - curr_time) / 60))
            print(disc_value_Ls)
            #########################################################################################################################################################################################################################################

    def est_density_ratio(self, h_dims=32, max_iter=100, batch_size=32,eps=0.01,print_freq=5):  # `omega_SA.py`
        """ step 4: learn \omega^{*\pi}() = d^pi_gamma() / d^b(), the state density ratio between the target and the behaviour policy
        w(S; S, A)
        """
        self.omegas = []

        for k in range(self.L):
            curr = now()
            train_ind = self.split_ind[k]["train_ind"]
            """ Q: other parameters? 
            Learning_rate = 1e-2, reg_weight = 0, n_layer = 2
            echo = False, batch_size = 64, max_iteration = 1e3, test_num = 0, epsilon = 1e-3
            """
            omega_func = omega_SASA_new.VisitationRatioModel_init_SASA(seed=self.seed,kernel=0,
                replay_buffer=sampler.SimpleReplayBuffer(trajs=[self.trajs[i] for i in train_ind]),
                target_policy=self.curr_policy[k], A_range=self.A_range, h_dims=h_dims,eps = eps)
            self.omega_func = omega_func
            omega_func.fit(batch_size=batch_size, gamma=self.gamma, max_iter=max_iter,print_freq = print_freq)
            self.omegas.append(omega_func.model)
            printG("<------ density ratio for fold {} DONE! Time cost = {:.1f} minutes -------->\n".format(k, (now() - curr) / 60))

    ##################################################
    def est_transition_density(self, v_dims=int(3), h_dims=int(32), epochs=100, batch_size=64, lr=0.0005, quantile=95,
                               trans_type='gauss', use_aitken='True'):
        """ step 5
        TODO: train_writer=None and optimizer
        """

        if trans_type == 'GAN':
            self.GANs = []
            for k in range(self.L):
                curr = now()
                train_ind = self.split_ind[k]["train_ind"]
                gan = GAN.learn_cond_sampler(trajs=[self.trajs[i] for i in train_ind], epochs=epochs, v_dims=v_dims,
                                             h_dims=h_dims
                                             , non_overlap_dim=self.non_overlap_dim, batch_size=batch_size
                                             , optimizer={"lr": lr, "w_clipping_norm": 1.0, "w_clipping_val": 0.5
                        , "gen_clipping_norm": 1.0, "gen_clipping_val": 0.5}, quantile=quantile)
                printG(
                    "<------ GAN (transition density) for fold {} DONE! Time cost = {:.1f} minutes -------->\n".format(
                        k, (now() - curr) / 60))
                self.GANs.append(gan)

        if trans_type == 'gauss':
            self.GANs = []
            for k in range(self.L):
                curr = now()
                train_ind = self.split_ind[k]["train_ind"]
                gan = gaussian.learn_cond_sampler(trajs=[self.trajs[i] for i in train_ind],num_A = self.num_A ,epochs=epochs,
                                                  v_dims=v_dims, h_dims=h_dims
                                                  , non_overlap_dim=self.non_overlap_dim, batch_size=batch_size
                                                  , optimizer={"lr": lr, "w_clipping_norm": 1.0, "w_clipping_val": 0.5
                        , "gen_clipping_norm": 1.0, "gen_clipping_val": 0.5}, use_aitken=use_aitken)
                printG("<------ gaussian (transition density) for fold {} DONE! Time cost = {:.1f} minutes -------->\n".format(k, (now() - curr) / 60))
                self.GANs.append(gan)

    ###################################################################################################################################################################################################################################################################################################################################################################################################################################################

    def get_lower_bound_components(self, NN=10, TT=20, M=10, G_ds_n_seeds=1e3, use_ori_trans=False):
        """ step 6: construct the objective function with real data and previous estimates.
        the order [fast -> slow] is different; TTNN, NT too
        TTNN -> first iter over T for each i, then iter over N

        Args:
            trajs: a list of trajs, each traj is a list of 4-tuple (S, A, R, S')

        Returns: a function w.r.t beta to be optimized
        """
        ######################## Init ########################
        self.NN, self.TT = NN, TT
        self.TTNN = NN * TT
        TN = self.N * self.T
        

        N = len(self.split_ind[0]["test_ind"])
        gamma, L = self.gamma, self.L

        FACTOR = (M * (1 - gamma ** TT)/ (1 - gamma ))
        gammas = arr([gamma ** t for t in range(TT)])
        broadcast_gammas = np.repeat(row_repeat(gammas, NN * TN, full_block=True), self.num_A, axis=1)
        broadcast_gammas_2 = np.repeat(np.expand_dims(np.tile(gammas, NN), 1), self.num_A, axis=1)

        gene_S_l1, gene_S_l1_one_old, gene_S_l2, gene_S_star_l2 = [], [], [], []
        old_probs_KL, term2_old_parts, term2_new_otherparts_with_gamma = [], [], []
        adv_l1_w_gamma = []
        adv_4_gene_S_star_l2, adv_4_gene_S_l2, adv_4_real_l2 = [], [], []
        l2_omega = []
        adv_next_S_l2, adv_S_l2 = [], []
        term2_news_omega_TD_gamma = []
        
        total_states = []

        ########################################################################
        ##################### calculate fixed data part ########################
        ########################################################################
        for k, split, G, old_value, omega in zip(range(self.L), self.split_ind.values(), self.GANs
                , self.curr_pi_value_funcs, self.omegas):
            """ training data has been used, test_idx is then used in this part to construct the functions """
            curr_time = now()

            
            ################## GENERATE SAMPLES ##################
            if use_ori_trans:
                states, actions, rewards, next_states = [
                    np.array([item[i] for traj in [self.trajs_ori[j] for j in split["test_ind"]] for item in traj]) for i in range(4)]
                TN_test = len(states)
                self.TN_test = TN_test
                self.TN = TN_test * self.L
                gene_S = self.ohio.sample_S_4VE(pi=self.curr_policy[k],NN=NN, TT=TT, seed = self.seed)
                gene_S = (gene_S - self.s_min)/(self.s_max - self.s_min)
                
                gene_cond_S, gene_cond_S_star = self.ohio.sample_cond_S_4VE(states, actions, next_states,self.curr_policy[k], M, TT ,seed = self.seed)
                gene_cond_S = (gene_cond_S - self.s_min)/(self.s_max - self.s_min)
                gene_cond_S_star = (gene_cond_S_star - self.s_min)/(self.s_max - self.s_min)
                
            else:
                states, actions, rewards, next_states = [
                    np.array([item[i] for traj in [self.trajs[j] for j in split["test_ind"]] for item in traj]) for i in range(4)]
                TN_test = len(states)
                self.TN_test = TN_test
                self.TN = TN_test * self.L
                gene_S = self.sample_S(self.init_sampler, self.curr_policy[k], G, NN, TT,self.S_dims)  # (TTNN, dim_S): shared within fold; can not shared across due to that G is different
                gene_cond_S, gene_cond_S_star = self.sample_cond_S(states, actions, next_states, M, TT, self.curr_policy[k], G)
            states, actions, rewards, next_states = [
                np.array([item[i] for traj in [self.trajs[j] for j in split["test_ind"]] for item in traj]) for i in range(4)]
            total_states.append(states)
            old_probs_KL.append(self.curr_policy[k].get_A_prob(gene_S, to_numpy=True))  # (TTNN, num_A)
            ####################################### term1: plug-in  #######################################
            adv_l1_w_gamma.append(old_value.A_func(gene_S) * broadcast_gammas_2)  # (TTNN, num_A)
            printG("<------  term 1 for fold {} DONE! Time cost = {:.1f} minutes -------->\n".format(k, (now() - curr_time) / 60))
            ################################# term2 (new) = l_1 (old) #######################################

            gene_S_l1.append(np.repeat(np.expand_dims(gene_S, 0), TN_test, 0))  # (TN, TTNN, dim_S)
            gene_S_l1_one_old.append(gene_S)  # (TTNN, dim_S)
            """ l1 term here; already gamma """
            term2_old_part, term2_new_omega_TD_gamma = self.l1_main_term(self.curr_policy[k], old_value, omega, gene_S,split)
            term2_old_parts.append(term2_old_part)  # scalar
            term2_news_omega_TD_gamma.append(term2_new_omega_TD_gamma)  # (TN, TTNN, num_A)
            printG("<------  term 2 for fold {} DONE! Time cost = {:.1f} minutes -------->\n".format(k, (now() - curr_time) / 60))
            ################################# term3 (new) = l_2 (old)#######################################
            #################### \tilde{omega}^*:  sample and average ####################
            # <------  sample for fold 0 DONE! Time cost = 0.5 minutes -------->
            # <------  3.1 for fold 0 DONE! Time cost = 4.5 minutes -------->
            # <------  3.2 for fold 0 DONE! Time cost = 8.6 minutes -------->
            l2_omega.append(self.l2_omega_term(G_ds_n_seeds, self.curr_policy[k], states, actions, omega))  # len = TN
            printG("<------  l2_omega_term for fold {} DONE! Time cost = {:.1f} minutes -------->\n".format(k, (
                        now() - curr_time) / 60))
            ################################################################################
            #### generate random samples; dim = (TN, M, TT, dim_S)

            gene_S_l2.append(gene_cond_S)
            gene_S_star_l2.append(gene_cond_S_star)
            ############################ 1; (TN, M, TT, num_A) ############################
            adv_next_S = old_value.A_func(gene_cond_S_star.reshape(-1, self.S_dims)).reshape(TN_test, M, TT,self.num_A)  / FACTOR
            adv_next_S_l2.append(np.stack([adv_next_S[:, :, i, :] * gammas[i] for i in range(len(gammas))], axis=2))
            printG("<------  3.1 for fold {} DONE! Time cost = {:.1f} minutes -------->\n".format(k, (now() - curr_time) / 60))
            ############################ 2; (TN, M, TT, num_A) ############################
            adv_S = old_value.A_func(gene_cond_S.reshape(-1, self.S_dims)).reshape(TN_test, M, TT, self.num_A) / FACTOR
            adv_S_l2.append(np.stack([adv_S[:, :, i, :] * gammas[i] for i in range(len(gammas))], axis=2))
            printG("<------  3.2 for fold {} DONE! Time cost = {:.1f} minutes -------->\n".format(k, (now() - curr_time) / 60))
            ############################ 3 ############################
            repeat_gammas = np.tile([gamma ** t for t in range(0, TN_test // N)], N)
#             adv_4_real_l2.append(
#                 np.dot(diag(repeat_gammas), old_value.A_func(states)))  # (TN, num_A); multiply with each row
            adv_4_real_l2.append(old_value.A_func(states))
            printG(
                "<------  lower bound construction for fold {} DONE! Time cost = {:.1f} minutes -------->\n".format(k, (now() - curr_time) / 60))
        #####################################################################################################

        self.components = {"adv_l1_w_gamma": arr(adv_l1_w_gamma)  # (L, TTNN, num_A)
                           # , "gene_S_l1" : np.vstack(gene_S_l1) # (TN, TTNN, dim_S)
            , "gene_S_l1_one_old": arr(gene_S_l1_one_old)  # (L, TTNN, dim_S)
            , "term2_news_omega_TD_gamma": arr(term2_news_omega_TD_gamma)  # (L, TTNN, num_A)
            , "term2_old_part": np.mean(term2_old_parts)
            , "gene_S_l2": np.vstack(gene_S_l2)  # (TN, M, TT, dim_S)
            , "total_states":np.vstack(total_states)
            , "gene_S_star_l2": np.vstack(gene_S_star_l2)  # (TN, M, TT, dim_S)
            , "adv_next_S_l2": np.vstack(adv_next_S_l2)  # (TN, M, TT, num_A)
            , "adv_S_l2": np.vstack(adv_S_l2)  # (TN, M, TT, num_A)
            , "adv_4_real_l2": np.vstack(adv_4_real_l2)  # (TN, num_A)
            , "l2_omega": np.concatenate(l2_omega)  # len = TN
            , "old_probs_KL": arr(old_probs_KL)  # (L, TTNN, num_A)
            , "broadcast_gamma_TTNN": np.tile(gammas, NN)  # (TTNN,)
                           }
        for key in self.components.keys():
            size = get_MB(self.components[key])
            if size > 10:
                print("size of {} = {:.0f} MB".format(key, size))

    ###################################################################################################################################################################################################################################################################################################################################################################################################################################################
    def construct_obj_fun(self, lamb=5.7,h_dims=32, use_init_paras=True, lr_lambda=0.0005
                          , Adam_para={"lr": 0.0005, "w_clipping_val": 0.5, "w_clipping_norm": 1.0}
                          , KL_delta=0.1,depth=1):
#         if use_init_paras:
#             if self.init_policy is "BEAR":
#                 actor = self.curr_policy.actor
#                 w = [actor.l1.weight.cpu().detach().numpy().T, actor.l1.bias.cpu().detach().numpy(),
#                      actor.l3.weight.cpu().detach().numpy().T, actor.l3.bias.cpu().detach().numpy()]
#             elif self.init_policy is "FQI":
#                 w = self.curr_policy.model.get_weights()
#             elif self.init_policy is "behav":
#                 w = self.curr_policy.model.get_weights()
#             elif self.init_policy is "last":
#                 w = self.curr_policy.get_weights()
#             init_paras = {"w1": w[0], "b1": w[1], "w2": w[2], "b2": w[3]}
#         else:
        init_paras = None

        self.KL_delta = KL_delta
        ######## Initialize the policy; should consider
        self.policy = policy.Policy(lamb = lamb, S_dims=self.S_dims, h_dims=h_dims, A_dims=self.A_dim, num_A=self.num_A,
                                    init_paras=init_paras, depth=depth)

        self.M = self.components["gene_S_star_l2"].shape[1]
        self.losses = []
        self.c_NT = np.sqrt(self.N * self.T)
        self.optimizer = tf.keras.optimizers.Adam(Adam_para["lr"], beta_1=Adam_para["beta"], clipnorm=Adam_para["w_clipping_norm"],clipvalue=Adam_para["w_clipping_val"])
#         self.optimizer_lambda = tf.keras.optimizers.Adam(lr_lambda, beta_1=0.5, clipnorm=Adam_para["w_clipping_norm"],
#                                                          clipvalue=Adam_para["w_clipping_val"])

    def improve_policy(self, max_iter=100, batch_size=32, print_freq=100, sd_G=3):
        self.iter += 1
        self.losses_lam = []
        self.KL_losses = []
        self.lams = []
        values = []
        self.lift = []
        self.l1 = []
        self.l2 = []
        self.l3 = []
        
        best_loss = 999999
        best_iter = 0
        total_states = self.components["total_states"]
        eval_state = total_states[:200]
        groups = {'value': ['loss', 'lift', 'True Values', 'l3'], 'KL': ['KL_loss', 'loglam']}
        plotlosses = PlotLosses(groups=groups)
        
        self.sample_counter = 0 
        for i in range(max_iter):
            ##########
            
            real_idx = self.lb_sample_buffer(batch_size)
            """
            should obtain two loss and do seperate optimization here. 
            why iterative? correct?
            """
            ##########
            with tf.GradientTape() as tape1:
                loss, lift ,l1 ,l2 , l3, kl_loss, lamb = self._compute_loss(real_idx)
#                 loss = -loss
            dw = tape1.gradient(loss, self.policy.parms)
#             print('gradient {}'.format(dw))
            self.optimizer.apply_gradients(zip(dw, self.policy.parms))

            ##########
            """ check. correct ??? """
#             with tf.GradientTape() as tape2:
#                 loss_lam, lift_1, KL_loss, l3 = self._compute_loss(real_idx)
#                 loss_lam = -loss_lam
#             dw = tape2.gradient(loss_lam, [self.policy.loglam])
#             self.optimizer_lambda.apply_gradients(zip(dw, [self.policy.loglam]))

            self.losses.append(loss.numpy().reshape(-1))  # for monitoring
#             self.l1.append(l1.numpy().reshape(-1))
#             self.l2.append(l2.numpy().reshape(-1))
#             self.l3.append(l3.numpy().reshape(-1))
            self.lift.append(lift)
            #             self.losses_lam.append(loss_lam.numpy()[0][0]) # for monitoring
            self.KL_losses.append(kl_loss)
            self.lams.append(self.policy.loglam.numpy().reshape(-1))
            if loss.numpy().reshape(-1)<best_loss:
##        wjn modified 5-6
#                 best_parms = [copy.deepcopy(x) for x in self.policy.parms]
# xd
                best_iter = i
                best_loss = loss.numpy().reshape(-1)
                
            ##########
            if i % print_freq == 0 and i >= print_freq:
                print('iter {},the best iteration is {} so far, corresponding loss is {}'.format(i,best_iter,best_loss))
                
                print('i :{} ,loss:{} ,lift:{}'.format(i,np.mean(self.losses[-1:]),np.mean(self.lift[-1:])))
                print('l1:{},l2:{},l3:{}'.format(l1.numpy(), l2.numpy(), l3.numpy()))
                #former idx (i-10):i
                print('kl loss :{}, lambda :{}'.format(np.mean(self.KL_losses[-1:]),np.mean(self.lams[-1:])))
#                 print('model parms {}'.format(self.policy.trainable_variables[0]))
                print('sample action under fix states {}'.format(self.policy.sample_A(eval_state)))
                discounted_values = self.eval_onpolicy(policy="improved", simulator="ohio", echo=False)
#                 print('now policy value is {}'.format(np.mean(discounted_values)))
                values.append(np.mean(discounted_values))
                
                    
#                 plotlosses.update({
#                     'loss': mean(self.losses[(i - 10):i]),
#                     'KL_loss': mean(self.KL_losses[(i - 10):i]),
#                     'loglam': mean(self.lams[(i - 10):i]),
#                     'True Values': np.mean(discounted_values),
#                     'lift': mean(self.lift[(i - 10):i]),
#                     'l3': mean(self.l3[(i - 10):i])
#                 })
#                 plotlosses.send()

##        wjn modified 5-6
#         for k,x in enumerate(self.policy.parms):
#             x.assign(best_parms[k])
# xd
            ############################################################################################################################################################################################################################################################################################################################################################

    def _compute_loss(self, real_idx=None):
        """ for a given target policy, calculate the loss func
        1. ratio according to batch size
            1. only sample NT, not NNTT; according to the definition.
            2. maybe we can avoid SGD
        """
        ######################## l1 ########################
        """ self.policy """
        """ no SGD; already average over NT """
        # (L, TTNN, num_A)
        A_prob_gene_S = self.policy.get_A_prob(self.components["gene_S_l1_one_old"], multi_dim=True)
        FACTOR = tf.reduce_sum(self.components["broadcast_gamma_TTNN"]) * self.L
        l1 = tf.reduce_sum(self.components["adv_l1_w_gamma"] * A_prob_gene_S) / FACTOR

        ######################## l2 ########################
        """ no SGD; already average over NT """
        l2_old_part = self.components["term2_old_part"]
        # (L, TTNN, num_A)
        l2_new_part = tf.reduce_sum(self.components["term2_news_omega_TD_gamma"] * A_prob_gene_S) / FACTOR / (1 - self.gamma)

        l2 = l2_new_part - l2_old_part

        ########################################################################
        ######################## l3: most memory-intense #######################
        ########################################################################
        """ self.policy """
        """ the two most memory-consuming parts """
        # (TN, M, TT, dim_S)
        
        states = self.components["total_states"][real_idx]
        A_prob_gene_S = self.policy.get_A_prob(self.components["gene_S_l2"][real_idx], multi_dim=True)
        # (TN, M, TT, dim_S)
        A_prob_gene_S_star = self.policy.get_A_prob(self.components["gene_S_star_l2"][real_idx], multi_dim=True)
        # (TN, num_A)
        A_prob_S = self.policy.get_A_prob(states)
        # (TN, )
        l3 = tf.reduce_sum(self.gamma * A_prob_gene_S_star * self.components["adv_next_S_l2"][real_idx] - \
                           A_prob_gene_S * self.components["adv_S_l2"][real_idx], (1, 2, 3)) + \
             (1 - self.gamma) * tf.reduce_sum(A_prob_S * self.components["adv_4_real_l2"][real_idx], 1)
        l3 = l3 * self.components["l2_omega"][real_idx] / (1 - self.gamma)  # (TN, )
        l3 = tf.reduce_mean(l3)

        ######################## KL term; (L, TTNN, num_A) ########################
        # (L, TTNN, num_A)
        KL_divergence = self.kl(curr_probs=self.components["old_probs_KL"]
                                , target_probs=self.policy.get_A_prob(self.components["gene_S_l1_one_old"],
                                                                      multi_dim=True))
        KL_loss = tf.tensordot(self.components["broadcast_gamma_TTNN"], KL_divergence, 1) * (1 - self.gamma) / (
                    self.NN * self.L * (1 - self.gamma ** self.TT))
        ################################# LOSS #################################
        lift = l1 + l2 + l3

        loss = -lift + tf.nn.relu(KL_loss - self.KL_delta) * tf.math.exp(self.policy.loglam)
        #         loss = -lift + tf.nn.relu(KL_loss - self.KL_delta)
#         printR(
#             'l1 term value -> {}, l2 term value -> {}, l3 term value ->{}, total value -> {}, constrain term -> {}'.format(
#                 l1, l2, l3, lift, KL_loss - self.KL_delta))
        return loss, lift, l1, l2, l3, KL_loss,tf.math.exp(self.policy.loglam)
    ##########################################################
    def lb_sample_buffer(self, batch_size=32, batch_size_KL=96):
        
        np.random.seed(self.seed + self.sample_counter)
        self.sample_counter += 1
        real_idx = choice(self.TN, batch_size, replace=True)  # ??
        ### for KL, do we need such sampling? ###
        sampled_idx = choice(self.TTNN, batch_size_KL, replace=True)
        return real_idx  # , sampled_idx

    def kl(self, curr_probs, target_probs):
        return tf.reduce_sum(curr_probs * tf.math.log(
            tf.cast(tf.clip_by_value(curr_probs, 1e-6, 1), tf.float64) / tf.cast(
                tf.clip_by_value(target_probs, 1e-6, 1), tf.float64)), (0, 2))

    ###################################################################################################################################################################################################################################################################################################################################################################################################################################################

    def eval_onpolicy(self, policy=None, simulator="ohio",  echo=True):
        """ for each policy, call the simulator to collect rewards, and calculate the values """
        if simulator is None:
            simulator = self.simulator
        if simulator == "ohio":
            
            if policy is not "true_behav" and policy is not "old" :
                if policy == "improved":
                    pi = self.policy

                elif policy == "est_behav":
                    pi = self.behav
                elif policy == "optimal":
                    pi = self.optimal_policy 
#                 print(pi)
                discounted_values, average_values = self.ohio_eval.eval_policy(pi=pi, gamma=self.gamma,seed=42)
                print('{} policy value is {}'.format(policy,np.mean(discounted_values)))
        
            elif policy == "old":
                values = []
                for k in range(self.L):
                    pi = self.curr_policy[k]
                    discounted_values, average_values = self.ohio_eval.eval_policy(pi=pi, gamma=self.gamma,seed=42)
                    values.append(discounted_values)
                    print('{} fold mean value is {}'.format(k,np.mean(discounted_values)))
                print('{} policy value is {}'.format(policy,np.mean(np.concatenate(values,axis=1))))
                discounted_values = np.concatenate(values,axis=1)
        
            else:
                # for behav, no need to evaluate the learned behav, can directly use the DGP to generate a lot!
                eval_trajs = self.ohio_eval.simu_one_seed(42)  # [N,T,SASA]
                discounted_values, average_values = self.ohio_eval.eval_trajs(eval_trajs, gamma=self.gamma)
                print('true behave policy value is {}'.format(np.mean(discounted_values)))
        return discounted_values        

#     def l1_main_term(self, curr_pi, curr_pi_value, omega, gene_S, split):
#         """ Between [].
#         gene_S: (TTNN, dim_S)
#         Out_shape = (TTNN * TN) *  num_A
#         """
#         states, actions, rewards, next_states = [
#             np.array([item[i] for traj in [self.trajs[j] for j in split["test_ind"]] for item in traj]) for i in
#             range(4)]
#         TN_test = len(states)
#         all_A = np.expand_dims(arr(list(self.A_range)), 1)
#         #################### Between {}. Out_shape = (TTNN * TN) *  num_A ####################
#         # w(states_r, states_t, actions_r); can be quite large; from fast to slow: (TN * TTNN * num_A, dim_S + dim_S_gene + dim_A) -> (TN * TTNN * num_A,)

#         #################### TD: real data. ####################
#         TD_error = rewards + self.gamma * curr_pi_value.V_func(next_states) - curr_pi_value.A_func(states,
#                                                                                                    actions) - curr_pi_value.V_func(
#             states)  # (TN,)

#         broadcast_TD = np.expand_dims(row_repeat(TD_error, self.TT * self.NN), 1)  # (TN * TTNN,)
#         ###########
#         repeat_gammas = np.tile([self.gamma ** t for t in range(0, self.TT)], TN_test * self.NN)

#         ###########
#         real_SA = np.hstack([states, np.expand_dims(actions, 1)])
#         # res = np.hstack([
#         #    np.tile(array1, (array2.shape[0], 1))
#         # , np.repeat(array2, array1.shape[0], axis=0)]
#         # )

#         omega_values = omega.predict_4_VE(
#             hstack_all_comb(hstack_all_comb(real_SA, gene_S), all_A))  # (TN * TTNN * num_A)
#         omega_values = omega_values.reshape(TN_test, self.TT * self.NN, self.num_A, order="F")  # (TN, TTNN, num_A)
#         omega_values = omega_values / np.mean(omega_values, axis=(1, 2), keepdims=1)
#         omega_values = omega_values.reshape(-1, self.num_A, order="F")  # (TN * TTNN, num_A)
#         ########### # (TTNN * TN, ); pi_old * omega_values ###########
#         A_prob_gene_S_old = curr_pi.get_A_prob(gene_S, to_numpy=True)  # (TTNN, num_A)
#         sum_A_prime_old = np.sum(row_repeat(A_prob_gene_S_old, TN_test, full_block=True) * omega_values, 1)

#         term2_old_part = np.sum(sum_A_prime_old * np.squeeze(broadcast_TD) * repeat_gammas) / \
#                          (1 - self.gamma) / sum(repeat_gammas)

#         ###########
#         term2_new_omega_TD_gamma = np.repeat(broadcast_TD, self.num_A, axis=1) * omega_values * np.repeat(
#             np.expand_dims(repeat_gammas, 1), self.num_A, axis=1)  # (TN * TTNN, num_A)
#         term2_new_omega_TD_gamma = term2_new_omega_TD_gamma.reshape(-1, self.TTNN, self.num_A)
#         # already average here!
#         term2_new_omega_TD_gamma = np.mean(term2_new_omega_TD_gamma, 0)
#         return term2_old_part, term2_new_omega_TD_gamma

#### modify by wjn 4-21############# 
    def l1_main_term(self, curr_pi, curr_pi_value, omega, gene_S, split):
        """ Between [].
        gene_S: (TTNN, dim_S)
        Out_shape = (TTNN * TN) *  num_A
        """
        states, actions, rewards, next_states = [
            np.array([item[i] for traj in [self.trajs[j] for j in split["test_ind"]] for item in traj]) for i in
            range(4)]
        TN_test = len(states)
        all_A = np.expand_dims(arr(list(self.A_range)), 1)
        #################### Between {}. Out_shape = (TTNN * TN) *  num_A ####################
        # w(states_r, states_t, actions_r); can be quite large; from fast to slow: (TN * TTNN * num_A, dim_S + dim_S_gene + dim_A) -> (TN * TTNN * num_A,)

        #################### TD: real data. ####################
        TD_error = rewards + self.gamma * curr_pi_value.V_func(next_states) - curr_pi_value.A_func(states,actions) - curr_pi_value.V_func(states)  # (TN,)
        TD_error = np.expand_dims(TD_error,1)

        broadcast_TD = row_repeat(TD_error, self.TT * self.NN,full_block=True) # (TN * TTNN,)
        ###########
        repeat_gammas = row_repeat(np.tile([self.gamma ** t for t in range(0, self.TT)],self.NN),TN_test)

        ###########
        real_SA = np.hstack([states, np.expand_dims(actions, 1)])
        # res = np.hstack([
        #    np.tile(array1, (array2.shape[0], 1))
        # , np.repeat(array2, array1.shape[0], axis=0)]
        # )

        omega_values = omega.predict_4_VE(
            hstack_all_comb(hstack_all_comb(real_SA, gene_S), all_A))  # (TN * TTNN * num_A)
        omega_values = omega_values.reshape(TN_test, self.TT * self.NN, self.num_A, order="F")  # (TN, TTNN, num_A)
        omega_values = omega_values / np.mean(omega_values, axis=(1, 2), keepdims=1)
        omega_values = omega_values.reshape(-1, self.num_A, order="F")  # (TN * TTNN, num_A)
        ########### # (TTNN * TN, ); pi_old * omega_values ###########
        A_prob_gene_S_old = curr_pi.get_A_prob(gene_S, to_numpy=True)  # (TTNN, num_A)
        sum_A_prime_old = np.sum(row_repeat(A_prob_gene_S_old, TN_test) * omega_values, 1)

        term2_old_part = np.sum(sum_A_prime_old * np.squeeze(broadcast_TD) * repeat_gammas) / \
                         (1 - self.gamma) / sum(repeat_gammas)

        ###########
        term2_new_omega_TD_gamma = np.repeat(broadcast_TD, self.num_A, axis=1) * omega_values * np.repeat(np.expand_dims(repeat_gammas, 1), self.num_A, axis=1)  # (TN * TTNN, num_A)
        term2_new_omega_TD_gamma = term2_new_omega_TD_gamma.reshape(-1, self.TTNN, self.num_A,order='F')
        # already average here!
        term2_new_omega_TD_gamma = np.mean(term2_new_omega_TD_gamma, 0)
        return term2_old_part, term2_new_omega_TD_gamma    

    def l2_omega_term(self, G_ds_n_seeds, old_pi, states, actions, omega):
        # can be splited
        init_S = self.init_sampler.sample(batch_size=int(G_ds_n_seeds))  # shared
        A_prob_sampled_S = old_pi.get_A_prob(init_S, to_numpy=True)  # (G_ds_n_seeds, num_A)
        ### NOTE: size ###
        real_SA = np.hstack([states, np.expand_dims(actions, 1)])
        all_A = np.expand_dims(arr(list(self.A_range)), 1)
        conc_S_A_gene_S = hstack_all_comb(hstack_all_comb(real_SA, init_S), all_A)
        omega_values = omega.predict_4_VE(
            conc_S_A_gene_S)  # from fast to slow: TN, G_ds_n_seeds, num_A. (TN * G_ds_n_seeds * num_A, )
        omega_values = omega_values.reshape(-1, int(G_ds_n_seeds), self.num_A,
                                            order="F")  # (TN, G_ds_n_seeds, num_A)
        omega_values = omega_values / np.mean(omega_values, axis=(1, 2), keepdims=1)

        sampled_ratio_term = arr([np.mean(np.sum(real_state * A_prob_sampled_S
                                                 , 1)) for real_state in omega_values])  # len = TN

        return sampled_ratio_term

    ################################################################################################################################################################################################################################################################################################################################################################################################################
    def sample_S(self, init_sampler, curr_pi, G, NN, TT, dim_S):
        gene_S = np.zeros((NN, TT, dim_S))
        S = init_sampler.sample(batch_size=NN)
        for t in range(TT):
            A = curr_pi.sample_A(S)
            if len(A.shape) == 1:
                A = np.expand_dims(A, 1)
            S = G.sample(S, A/4)  # GAN
            gene_S[:, t] = S.copy()
        gene_S = gene_S.reshape(-1, dim_S)  # stack gene_S; (TTNN, dim_S)
        return gene_S

    def sample_cond_S(self, states, actions, next_states
                      , M, TT, curr_pi, G):
        """
        S: (S,A)
        S^*: (S, pi(S))
        """
        TN = int(len(states))
        dim_S = states.shape[1]
        ##############
        gene_S = np.zeros((TN * M, TT, dim_S))
        # dim = (TN * M, TT, dim_S)
        S = row_repeat(states, M, full_block=True)  # True -> M is slower
        gene_S[:, 0] = S.copy()
        A = np.tile(actions, M)
        for t in range(1, TT - 1):
            if len(A.shape) == 1:
                A = np.expand_dims(A, 1)
            S = G.sample(S, A/4)
            gene_S[:, t] = S.copy()
            A = curr_pi.sample_A(S)
        ###################################
        gene_S_star = np.zeros((TN * M, TT, dim_S))
        S = row_repeat(next_states, M, full_block=True)  # (TN * M, dim_S)
        A = curr_pi.sample_A(S)
        gene_S_star[:, 0] = S
        for t in range(1, TT - 1):
            if len(A.shape) == 1:
                A = np.expand_dims(A, 1)
            S = G.sample(S, A/4)
            A = curr_pi.sample_A(S)
            gene_S_star[:, t] = S.copy()
        # dim = (TN, M, TT, dim_S)
        gene_S = gene_S.reshape(TN, M, TT, dim_S,order='F')
        gene_S_star = gene_S_star.reshape(TN, M, TT, dim_S,order='F')
        return [gene_S, gene_S_star]

    ##############################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################

#     def _normalize(self, weights, batch_size):
#         weights = tf.reshape(weights, [batch_size, batch_size])
#         weights_sum = tf.math.reduce_sum(weights, axis=1, keepdims=True) + 1e-6
#         weights = weights / weights_sum
#         return tf.reshape(weights, [batch_size**2])


