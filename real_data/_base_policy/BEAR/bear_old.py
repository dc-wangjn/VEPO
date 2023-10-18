import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

# from logger import create_stats_ordered_dict
from logger import logger
# torch.set_default_tensor_type('torch.cuda.FloatTensor')

###########################################################################################################################################################################################################################################################################################################################

class BEAR(object):
    def __init__(self, num_qs = 2, state_dim = 3, action_dim = 1, latent_dim = 32, num_A = 5, delta_conf=0.1, use_bootstrap=True, version=0, lambda_=0.4,
                 threshold=0.05, mode='auto', num_samples_match=10, mmd_sigma=10.0,
                 lagrange_thresh=10.0, use_kl=False, use_ensemble=True, kernel_type='laplacian',seed=0):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")

        self.actor = RegularActor(state_dim, action_dim, latent_dim, num_A, 1).to(self.device)
        self.actor_target = RegularActor(state_dim, action_dim, latent_dim, num_A, 1).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = EnsembleCritic(num_qs, state_dim, action_dim).to(self.device)
        self.critic_target = EnsembleCritic(num_qs, state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.vae = VAE(state_dim, action_dim, latent_dim, num_A).to(self.device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters()) 

        self.num_A = num_A
        self.action_dim = action_dim
        self.delta_conf = delta_conf
        self.use_bootstrap = use_bootstrap
        self.version = version
        self._lambda = lambda_
        self.threshold = threshold
        self.mode = mode
        self.num_qs = num_qs
        self.num_samples_match = num_samples_match
        self.mmd_sigma = mmd_sigma
        self.lagrange_thresh = lagrange_thresh
        self.use_kl = use_kl
        self.use_ensemble = use_ensemble
        self.kernel_type = kernel_type
        torch.autograd.set_detect_anomaly(True)
        torch.cuda.manual_seed_all(seed)#wjn5-12modified
        np.random.seed(seed)#wjn5-12modified
        if self.mode == 'auto':
            # Use lagrange multipliers on the constraint if set to auto mode 
            # for the purpose of maintaing support matching at all times
            self.log_lagrange2 = torch.randn((), requires_grad=True, device=self.device)
            self.lagrange2_opt = torch.optim.Adam([self.log_lagrange2,], lr=1e-3)

        self.epoch = 0

    def mmd_loss_laplacian(self, samples1, samples2, sigma=0.2):
        """MMD constraint with Laplacian kernel for support matching"""
        # sigma is set to 10.0 for hopper, cheetah and 20 for walker/ant
        diff_x_x = samples1.unsqueeze(2) - samples1.unsqueeze(1)  # B x N x N x d
        diff_x_x = torch.mean((-(diff_x_x.abs()).sum(-1)/(2.0 * sigma)).exp(), dim=(1,2))

        diff_x_y = samples1.unsqueeze(2) - samples2.unsqueeze(1)
        diff_x_y = torch.mean((-(diff_x_y.abs()).sum(-1)/(2.0 * sigma)).exp(), dim=(1, 2))

        diff_y_y = samples2.unsqueeze(2) - samples2.unsqueeze(1)  # B x N x N x d
        diff_y_y = torch.mean((-(diff_y_y.abs()).sum(-1)/(2.0 * sigma)).exp(), dim=(1,2))

        overall_loss = (diff_x_x + diff_y_y - 2.0 * diff_x_y + 1e-6).sqrt()
        return overall_loss
    
    def mmd_loss_gaussian(self, samples1, samples2, sigma=0.2):
        """MMD constraint with Gaussian Kernel support matching"""
        # sigma is set to 10.0 for hopper, cheetah and 20 for walker/ant
        diff_x_x = samples1.unsqueeze(2) - samples1.unsqueeze(1)  # B x N x N x d
        diff_x_x = torch.mean((-(diff_x_x.pow(2)).sum(-1)/(2.0 * sigma)).exp(), dim=(1,2))

        diff_x_y = samples1.unsqueeze(2) - samples2.unsqueeze(1)
        diff_x_y = torch.mean((-(diff_x_y.pow(2)).sum(-1)/(2.0 * sigma)).exp(), dim=(1, 2))

        diff_y_y = samples2.unsqueeze(2) - samples2.unsqueeze(1)  # B x N x N x d
        diff_y_y = torch.mean((-(diff_y_y.pow(2)).sum(-1)/(2.0 * sigma)).exp(), dim=(1,2))

        overall_loss = (diff_x_x + diff_y_y - 2.0 * diff_x_y + 1e-6).sqrt()
        return overall_loss

    def kl_loss(self, samples1, state, sigma=0.2):
        """We just do likelihood, we make sure that the policy is close to the
           data in terms of the KL."""
        state_rep = state.unsqueeze(1).repeat(1, samples1.size(1), 1).view(-1, state.size(-1))
        samples1_reshape = samples1.view(-1, samples1.size(-1))
        samples1_log_pis = self.actor.log_pis(state=state_rep, raw_action=samples1_reshape)
        samples1_log_prob = samples1_log_pis.view(state.size(0), samples1.size(1))
        return (-samples1_log_prob).mean(1)
    
    def entropy_loss(self, samples1, state, sigma=0.2):
        state_rep = state.unsqueeze(1).repeat(1, samples1.size(1), 1).view(-1, state.size(-1))
        samples1_reshape = samples1.view(-1, samples1.size(-1))
        samples1_log_pis = self.actor.log_pis(state=state_rep, raw_action=samples1_reshape)
        samples1_log_prob = samples1_log_pis.view(state.size(0), samples1.size(1))
        samples1_prob = samples1_log_prob.clamp(min=-5, max=4).exp()
        return (samples1_prob).mean(1)
    
    ################################################################################################################################
    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005):
        for it in range(iterations):
            state_np, next_state_np, action, reward = replay_buffer.sample(batch_size)
            state           = torch.FloatTensor(state_np).to(self.device)
            action          = torch.FloatTensor(action).to(self.device)
            next_state      = torch.FloatTensor(next_state_np).to(self.device)
            reward          = torch.FloatTensor(reward).to(self.device)
            
            # Train the Behaviour cloning policy to be able to take more than 1 sample for MMD
            """ ??? """
            recon = self.vae(state, action)
            recon_loss = F.mse_loss(recon, action)
            # KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            vae_loss = recon_loss #+ 0.5 * KL_loss

            self.vae_optimizer.zero_grad() # ???
            vae_loss.backward()
            self.vae_optimizer.step()

            #############################################
            # Critic Training: In this step, we explicitly compute the actions  
            ##########################################
            with torch.no_grad():
                # Duplicate state 10 times (10 is a hyperparameter chosen by BCQ)
                state_rep = torch.FloatTensor(np.repeat(next_state_np, 10, axis=0)).to(self.device)
                
                # Compute value of perturbed actions sampled from the VAE
                target_Qs = self.critic_target(state_rep, self.actor_target(state_rep))

                # Soft Clipped Double Q-learning 
                lam = 0.75
                target_Q = lam * target_Qs.min(0)[0] + (1 - lam) * target_Qs.max(0)[0]
                target_Q = target_Q.view(batch_size, -1).max(1)[0].view(-1, 1)
                # target_Q = reward + done * discount * target_Q
                target_Q = reward + discount * target_Q

            current_Qs = self.critic(state, action, with_var=False)
            if self.use_bootstrap: 
                critic_loss = (F.mse_loss(current_Qs[0], target_Q, reduction='none')).mean() +\
                            (F.mse_loss(current_Qs[1], target_Q, reduction='none')).mean() 
            else:
                critic_loss = F.mse_loss(current_Qs[0], target_Q) + F.mse_loss(current_Qs[1], target_Q) 

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            #############################################
            # Action Training
            ##########################################
            
            # If you take less samples (but not too less, else it becomes statistically inefficient), it is closer to a uniform support set matching
            num_samples = self.num_samples_match
            sampled_actions = self.vae.decode_multiple(state, num_decode=num_samples)  # B x N x d
            actor_actions = self.actor.sample_multiple(state, num_samples)#  num)
            
            # MMD done on raw actions (before tanh), to prevent gradient dying out due to saturation
            """ KL """
            if self.use_kl:
                mmd_loss = self.kl_loss(sampled_actions, state)
            else:
                if self.kernel_type == 'gaussian':
                    mmd_loss = self.mmd_loss_gaussian(sampled_actions, actor_actions, sigma=self.mmd_sigma)
                else:
                    mmd_loss = self.mmd_loss_laplacian(sampled_actions, actor_actions, sigma=self.mmd_sigma)

            action_divergence = ((sampled_actions - actor_actions)**2).sum(-1)

            ######################## Update through TD3 style ########################
            critic_qs, std_q = self.critic.q_all(state, actor_actions[:, 0, :], with_var=True)
            critic_qs = self.critic.q_all(state.unsqueeze(0).repeat(num_samples, 1, 1).view(num_samples*state.size(0), state.size(1)), actor_actions.permute(1, 0, 2).contiguous().view(num_samples*actor_actions.size(0), actor_actions.size(2)))
            critic_qs = critic_qs.view(self.num_qs, num_samples, actor_actions.size(0), 1)
            critic_qs = critic_qs.mean(1)
            """ std_q here ! """
            std_q = torch.std(critic_qs, dim=0, keepdim=False, unbiased=False)

            if not self.use_ensemble:
                std_q = torch.zeros_like(std_q).to(self.device)
                
            if self.version == '0':
                critic_qs = critic_qs.min(0)[0]
            elif self.version == '1':
                critic_qs = critic_qs.max(0)[0]
            elif self.version == '2':
                critic_qs = critic_qs.mean(0)
            
            ############ Actor Loss ########################
            # We do support matching with a warmstart which happens to be reasonable around epoch 20 during training
            """ KL-loss also here! 
            min loss = max -loss
            """
            if self.epoch >= 20: 
                if self.mode == 'auto':
                    actor_loss = (-critic_qs +\
                        self._lambda * (np.sqrt((1 - self.delta_conf)/self.delta_conf)) * std_q +\
                        self.log_lagrange2.exp() * mmd_loss
                                 ).mean()
                else:
                    actor_loss = (-critic_qs +\
                        self._lambda * (np.sqrt((1 - self.delta_conf)/self.delta_conf)) * std_q +\
                        100.0*mmd_loss).mean()      
                    # This coefficient is hardcoded, and is different for different tasks. I would suggest using auto, as that is the one used in the paper and works better.
            else:
                if self.mode == 'auto':
                    actor_loss = (self.log_lagrange2.exp() * mmd_loss).mean()
                else:
                    actor_loss = 100.0*mmd_loss.mean()
                    
            ################################################
            std_loss = self._lambda * (np.sqrt((1 - self.delta_conf) / self.delta_conf)) * std_q.detach() 

            self.actor_optimizer.zero_grad()
            if self.mode =='auto':
                actor_loss.backward(retain_graph=True)
            else:
                actor_loss.backward()
            # torch.nn.utils.clip_grad_norm(self.actor.parameters(), 10.0)
            self.actor_optimizer.step()
            ################## MULTIPLIER ##################
            # Threshold for the lagrange multiplier
            if self.use_kl:
                thresh = -2.0
            else:
                thresh = 0.05


            if self.mode == 'auto':
                # negative!
                # for this line, all these parameters are fixed, except for std_q; purpose for this line? correct the first one?
                lagrange_loss = (-critic_qs +\
                        self._lambda * (np.sqrt((1 - self.delta_conf)/self.delta_conf)) * (std_q) + self.log_lagrange2.exp() * (mmd_loss - thresh) # why threshold here?
                                ).mean()

                self.lagrange2_opt.zero_grad()
                (-lagrange_loss).backward()
                # self.lagrange1_opt.step()
                self.lagrange2_opt.step() 
                self.log_lagrange2.data.clamp_(min = -5.0, max = self.lagrange_thresh) # 10
            
            ### Update Target Networks ##########################################
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                    
    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
    """ TODO """
    """    def select_action(self, state):      
        # When running the actor, we just select action based on the max of the Q-function computed over
            samples from the policy -- which biases things to support.
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).repeat(10, 1).to(device)
            action = self.actor(state)
            q1 = self.critic.q1(state, action)
            ind = q1.max(0)[1]
        return action[ind].cpu().data.numpy().flatten()
    """
    def get_A_prob(self, states, actions = None, to_numpy= False):
        """
        probability for one action / prob matrix
        ; for NN
        sometimes prediction and sometimes training
        """
        with torch.no_grad():
            
            A_prob = self.actor(torch.FloatTensor(states).to(self.device), prob_only = True)
            if to_numpy:
                A_prob = A_prob.cpu().detach().numpy()
            if actions is None:
                return A_prob
            else:
                return A_prob[range(len(actions)), actions.astype(int)]
            
    def get_greedy_A(self, states, to_numpy= True):
        """ most possible action
        """
        with torch.no_grad():
            if to_numpy:
                return self.actor(torch.FloatTensor(states).to(self.device), prob_only = True).cpu().detach().numpy().argmax(1)
    
    def sample_A(self, states, to_numpy= True):
        # do we really need seeds?
        # different states (each row) -> diff prob -> diff sampled actions
        # np.random.seed(self.sample_A_seed)
        # self.sample_A_seed += 1
        with torch.no_grad():
            if to_numpy:
                return self.actor(torch.FloatTensor(states).to(self.device)).cpu().detach().numpy()
##################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
        # Do all logging here
#         logger.record_dict(create_stats_ordered_dict(
#             'Q_target',
#             target_Q.cpu().data.numpy(),
#         ))
#         if self.mode == 'auto':
#             # logger.record_tabular('Lagrange1', self.log_lagrange1.exp().cpu().data.numpy())
#             logger.record_tabular('Lagrange2', self.log_lagrange2.exp().cpu().data.numpy())

#         logger.record_tabular('Actor Loss', actor_loss.cpu().data.numpy())
#         logger.record_tabular('Critic Loss', critic_loss.cpu().data.numpy())
#         logger.record_tabular('Std Loss', std_loss.cpu().data.numpy().mean())
#         logger.record_dict(create_stats_ordered_dict(
#             'MMD Loss',
#             mmd_loss.cpu().data.numpy()
#         ))
#         logger.record_dict(create_stats_ordered_dict(
#             'Sampled Actions',
#             sampled_actions.cpu().data.numpy()
#         ))
#         logger.record_dict(create_stats_ordered_dict(
#             'Actor Actions',
#             actor_actions.cpu().data.numpy()
#         ))
#         logger.record_dict(create_stats_ordered_dict(
#             'Current_Q',
#             current_Qs.cpu().data.numpy()
#         ))
#         logger.record_dict(create_stats_ordered_dict(
#             'Action_Divergence',
#             action_divergence.cpu().data.numpy()
#         ))
#         logger.record_dict(create_stats_ordered_dict(
#             'Raw Action_Divergence',
#             raw_action_divergence.cpu().data.numpy()
#         ))
        self.epoch = self.epoch + 1

###########################################################################################################################################################################################################################################################################################################################


def atanh(x):# log[(1+x)/(1-x)] / 2
    one_plus_x = (1 + x).clamp(min=1e-7)
    one_minus_x = (1 - x).clamp(min=1e-7)
    return 0.5*torch.log(one_plus_x/ one_minus_x)

""" continuous actions. hidden dim is too large.
"""
class RegularActor(nn.Module):
    """A probabilistic actor which does regular stochastic mapping of actions from states"""
    def __init__(self, state_dim = 15, action_dim = 1, latent_dim = 32, num_A = 5, depth = 1):
        super(RegularActor, self).__init__()
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.depth = depth
        if self.depth == 1:
            self.l1 = nn.Linear(state_dim, latent_dim)
            self.l3 = nn.Linear(latent_dim, num_A)
        elif self.depth == 2:
            self.l1 = nn.Linear(state_dim, latent_dim)
            self.l2 = nn.Linear(latent_dim, latent_dim)
            self.l3 = nn.Linear(latent_dim, num_A)
        self.latent_dim = latent_dim
        self.num_A = num_A
    
    """ Q: how to link with our methods? """
    def forward(self, state, prob_only = False):
        a = F.relu(self.l1(state))
        if self.depth == 2:
            a = F.relu(self.l2(a))
        A_prob = F.softmax(self.l3(a))
        if prob_only:
            return A_prob
        else:
            sample_As = (A_prob.cumsum(1) > torch.FloatTensor(np.random.rand(A_prob.size(0))[:,None]).to(self.device)).double().argmax(1)
            return sample_As.unsqueeze_(-1).to(torch.float32) 

        # This function doesn’t work directly with NLLLoss, which expects the Log to be computed between the Softmax and itself. Use log_softmax instead (it’s faster and has better numerical properties).
        
    def sample_multiple(self, state, num_sample=10):
        a = F.relu(self.l1(state))
        if self.depth == 2:
            a = F.relu(self.l2(a))
        A_prob = F.softmax(self.l3(a))
        sampled_thre = torch.FloatTensor(np.random.rand(A_prob.size(0), num_sample))
        sample_As = torch.stack([(A_prob.cumsum(1) > sampled_thre[:, i].unsqueeze_(-1).to(self.device)).double().argmax(1) for i in range(num_sample)], 1)
        return sample_As.unsqueeze_(-1).to(torch.float32) 
    
    def log_pis(self, state, action=None, raw_action=None):
        """Get log pis for the model."""
        a = F.relu(self.l1(state))
        if self.depth == 2:
            a = F.relu(self.l2(a))
        A_prob = F.softmax(self.l3(a))
        return torch.log(A_prob)
    
###########################################################################################################################################################################################################################################################################################################################
class Critic(nn.Module):
    """Regular critic used in off-policy RL"""
    def __init__(self, state_dim = 3, action_dim = 1, latent_dim = 10):
        super(Critic, self).__init__()
        self.latent_dim = latent_dim
        self.l1 = nn.Linear(state_dim + action_dim, latent_dim)
        self.l2 = nn.Linear(latent_dim, latent_dim)
        self.l3 = nn.Linear(latent_dim, 1)

        self.l4 = nn.Linear(state_dim + action_dim, latent_dim)
        self.l5 = nn.Linear(latent_dim, latent_dim)
        self.l6 = nn.Linear(latent_dim, 1)

    def forward(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(torch.cat([state, action], 1)))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def q1(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

class EnsembleCritic(nn.Module):
    """ Critic which does have a network of 4 Q-functions"""
    def __init__(self, num_qs = 2, state_dim = 3, action_dim = 1, latent_dim = 10):
        super(EnsembleCritic, self).__init__()
        
        self.num_qs = num_qs

        self.l1 = nn.Linear(state_dim + action_dim, latent_dim)
        self.l2 = nn.Linear(latent_dim, latent_dim)
        self.l3 = nn.Linear(latent_dim, 1)

        self.l4 = nn.Linear(state_dim + action_dim, latent_dim)
        self.l5 = nn.Linear(latent_dim, latent_dim)
        self.l6 = nn.Linear(latent_dim, 1)

    def forward(self, state, action, with_var=False):
        all_qs = []
        
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(torch.cat([state, action], 1)))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        all_qs = torch.cat(
            [q1.unsqueeze(0), q2.unsqueeze(0),], 0) # q3.unsqueeze(0), q4.unsqueeze(0)], 0)   # Num_q x B x 1
        if with_var:
            std_q = torch.std(all_qs, dim=0, keepdim=False, unbiased=False)
            return all_qs, std_q
        return all_qs

    def q1(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1
    
    def q_all(self, state, action, with_var=False):
        all_qs = []
        
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(torch.cat([state, action], 1)))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        all_qs = torch.cat(
            [q1.unsqueeze(0), q2.unsqueeze(0),], 0) # q3.unsqueeze(0), q4.unsqueeze(0)], 0)  # Num_q x B x 1
        if with_var:
            std_q = torch.std(all_qs, dim=0, keepdim=False, unbiased=False)
            return all_qs, std_q
        return all_qs
###########################################################################################################################################################################################################################################################################################################################
# Vanilla Variational Auto-Encoder 
class VAE(nn.Module):
    """VAE Based behavior cloning also used in Fujimoto et.al. (ICML 2019)"""
    def __init__(self, state_dim = 3, action_dim = 1, latent_ec_dim = 10, latent_dim = 10, num_A = 5):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        super(VAE, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, latent_ec_dim)
        self.e2 = nn.Linear(latent_ec_dim, latent_dim)

#         self.mean = nn.Linear(latent_ec_dim, latent_dim)
#         self.log_std = nn.Linear(latent_ec_dim, latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, latent_ec_dim)
        self.d2 = nn.Linear(latent_ec_dim, latent_ec_dim)
        self.d3 = nn.Linear(latent_ec_dim, int(num_A))
        self.latent_ec_dim, self.state_dim, self.action_dim = latent_ec_dim, state_dim, action_dim
        self.num_A = num_A
        self.latent_dim = latent_dim

    def forward(self, state, action): # encorder
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))
        u = self.decode(state, z)
        return u
    
    def decode(self, state, z=None):
        if z is None:
            z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), self.latent_dim))).to(self.device).clamp(-0.5, 0.5)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        A_prob = F.softmax(self.d3(a)) #, self.d3(a)
        return A_prob
        #sample_As = (A_prob.cumsum(1) > torch.FloatTensor(np.random.rand(A_prob.size(0))[:,None])).to(torch.float32).argmax(1)
        #return sample_As.unsqueeze_(-1).to(torch.float32) 
    
    """ Q: appropriate? """
    def decode_multiple(self, state, z=None, num_decode=12):
        if z is None:
            z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), num_decode, self.latent_dim))).to(self.device).clamp(-0.5, 0.5)

        a = F.relu(self.d1(torch.cat([state.unsqueeze(0).repeat(num_decode, 1, 1).permute(1, 0, 2), z], 2)))
        a = F.relu(self.d2(a))
        A_prob = F.softmax(self.d3(a)) #, self.d3(a)
        return A_prob
        # sampled_thre = torch.FloatTensor(np.random.rand(A_prob.size(0), num_sample))
        # sample_As = torch.stack([(A_prob.cumsum(1) > sampled_thre[:, i].unsqueeze_(-1)).to(torch.float32).argmax(1) for i in range(num_sample)], 1)
        # return sample_As.unsqueeze_(-1).to(torch.float32) 
    #self.num_A * torch.tanh(self.d3(a)), self.d3(a)
    

        
##############################################################################################################################################################################################################################################################################################################################
def weighted_mse_loss(inputs, target, weights):
    return torch.mean(weights * (inputs - target)**2)




##############################################################################################################################################################################################################################################################################################################################


#     def decode_softplus(self, state, z=None):
#         if z is None:
#             z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), self.latent_dim))).to(device).clamp(-0.5, 0.5)
        
#         a = F.relu(self.d1(torch.cat([state, z], 1)))
#         a = F.relu(self.d2(a))
        
    
#     def decode_bc(self, state, z=None):
#         if z is None:
#                 z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), self.latent_dim))).to(device)

#         a = F.relu(self.d1(torch.cat([state, z], 1)))
#         a = F.relu(self.d2(a))
#         return self.num_A * torch.tanh(self.d3(a))

#     def decode_bc_test(self, state, z=None):
#         if z is None:
#                 z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), self.latent_dim))).to(device).clamp(-0.25, 0.25)

#         a = F.relu(self.d1(torch.cat([state, z], 1)))
#         a = F.relu(self.d2(a))
#         return self.num_A * torch.tanh(self.d3(a))