import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    """Actor used in BCQ"""
    def __init__(self, state_dim, action_dim, max_action, threshold=0.05):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        
        self.max_action = max_action
        self.threshold = threshold

    def forward(self, state, action):
        a = F.relu(self.l1(torch.cat([state, action], 1)))
        a = F.relu(self.l2(a))
        a = self.threshold * self.max_action * torch.tanh(self.l3(a))
        return (a + action).clamp(-self.max_action, self.max_action)

class ActorTD3(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(ActorTD3, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, x, preval=False):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        pre_tanh_val = x
        x = self.max_action * torch.tanh(self.l3(x))
        if not preval:
            return x
        return x, pre_tanh_val

def atanh(x):
    one_plus_x = (1 + x).clamp(min=1e-7)
    one_minus_x = (1 - x).clamp(min=1e-7)
    return 0.5*torch.log(one_plus_x/ one_minus_x)

class RegularActor(nn.Module):
    """A probabilistic actor which does regular stochastic mapping of actions from states"""
    def __init__(self, state_dim, action_dim, max_action,):
        super(RegularActor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.mean = nn.Linear(300, action_dim)
        self.log_std = nn.Linear(300, action_dim)
        self.max_action = max_action
    
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mean_a = self.mean(a)
        log_std_a = self.log_std(a)
        
        std_a = torch.exp(log_std_a)
        z = mean_a + std_a * torch.FloatTensor(np.random.normal(0, 1, size=(std_a.size()))).to(device) 
        return self.max_action * torch.tanh(z)

    def sample_multiple(self, state, num_sample=10):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mean_a = self.mean(a)
        log_std_a = self.log_std(a)
        
        std_a = torch.exp(log_std_a)
        # This trick stabilizes learning (clipping gaussian to a smaller range)
        z = mean_a.unsqueeze(1) +\
             std_a.unsqueeze(1) * torch.FloatTensor(np.random.normal(0, 1, size=(std_a.size(0), num_sample, std_a.size(1)))).to(device).clamp(-0.5, 0.5)
        return self.max_action * torch.tanh(z), z 

    def log_pis(self, state, action=None, raw_action=None):
        """Get log pis for the model."""
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mean_a = self.mean(a)
        log_std_a = self.log_std(a)
        std_a = torch.exp(log_std_a)
        normal_dist = td.Normal(loc=mean_a, scale=std_a, validate_args=True)
        if raw_action is None:
            raw_action = atanh(action)
        else:
            action = torch.tanh(raw_action)
        log_normal = normal_dist.log_prob(raw_action)
        log_pis = log_normal.sum(-1)
        log_pis = log_pis - (1.0 - action**2).clamp(min=1e-6).log().sum(-1)
        return log_pis

class Critic(nn.Module):
    """Regular critic used in off-policy RL"""
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

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
    def __init__(self, num_qs, state_dim, action_dim):
        super(EnsembleCritic, self).__init__()
        
        self.num_qs = num_qs

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

        # self.l7 = nn.Linear(state_dim + action_dim, 400)
        # self.l8 = nn.Linear(400, 300)
        # self.l9 = nn.Linear(300, 1)

        # self.l10 = nn.Linear(state_dim + action_dim, 400)
        # self.l11 = nn.Linear(400, 300)
        # self.l12 = nn.Linear(300, 1)

    def forward(self, state, action, with_var=False):
        all_qs = []
        
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(torch.cat([state, action], 1)))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        # q3 = F.relu(self.l7(torch.cat([state, action], 1)))
        # q3 = F.relu(self.l8(q3))
        # q3 = self.l9(q3)

        # q4 = F.relu(self.l10(torch.cat([state, action], 1)))
        # q4 = F.relu(self.l11(q4))
        # q4 = self.l12(q4)

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

        # q3 = F.relu(self.l7(torch.cat([state, action], 1)))
        # q3 = F.relu(self.l8(q3))
        # q3 = self.l9(q3)

        # q4 = F.relu(self.l10(torch.cat([state, action], 1)))
        # q4 = F.relu(self.l11(q4))
        # q4 = self.l12(q4)

        all_qs = torch.cat(
            [q1.unsqueeze(0), q2.unsqueeze(0),], 0) # q3.unsqueeze(0), q4.unsqueeze(0)], 0)  # Num_q x B x 1
        if with_var:
            std_q = torch.std(all_qs, dim=0, keepdim=False, unbiased=False)
            return all_qs, std_q
        return all_qs

# Vanilla Variational Auto-Encoder 
class VAE(nn.Module):
    """VAE Based behavior cloning also used in Fujimoto et.al. (ICML 2019)"""
    def __init__(self, state_dim, action_dim, latent_dim, max_action):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, 750)
        self.e2 = nn.Linear(750, 750)

        self.mean = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, 750)
        self.d2 = nn.Linear(750, 750)
        self.d3 = nn.Linear(750, action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim


    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability 
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.FloatTensor(np.random.normal(0, 1, size=(std.size()))).to(device) 
        
        u = self.decode(state, z)

        return u, mean, std
    
    def decode_softplus(self, state, z=None):
        if z is None:
            z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), self.latent_dim))).to(device).clamp(-0.5, 0.5)
        
        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        
    def decode(self, state, z=None):
        if z is None:
                z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), self.latent_dim))).to(device).clamp(-0.5, 0.5)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))
    
    def decode_bc(self, state, z=None):
        if z is None:
                z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), self.latent_dim))).to(device)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))

    def decode_bc_test(self, state, z=None):
        if z is None:
                z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), self.latent_dim))).to(device).clamp(-0.25, 0.25)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))
    
    def decode_multiple(self, state, z=None, num_decode=10):
        """Decode 10 samples atleast"""
        if z is None:
            z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), num_decode, self.latent_dim))).to(device).clamp(-0.5, 0.5)

        a = F.relu(self.d1(torch.cat([state.unsqueeze(0).repeat(num_decode, 1, 1).permute(1, 0, 2), z], 2)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a)), self.d3(a)

    
class BCQ(object):
    def __init__(self, state_dim, action_dim, max_action, cloning=False):
        latent_dim = action_dim * 2

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.vae = VAE(state_dim, action_dim, latent_dim, max_action).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters()) 

        self.max_action = max_action
        self.action_dim = action_dim
        self.use_cloning = cloning


    def select_action(self, state):
        if self.use_cloning:
            return self.select_action_cloning(state)       
        with torch.no_grad():
                state = torch.FloatTensor(state.reshape(1, -1)).repeat(10, 1).to(device)
                action = self.actor(state, self.vae.decode(state))
                q1 = self.critic.q1(state, action)
                ind = q1.max(0)[1]
#                 print()
        return action[ind].cpu().data.numpy().flatten()

    def select_action_cloning(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            action = self.vae.decode_bc_test(state)
        return action[0].cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005):
        for it in range(iterations):
            # print ('Iteration : ', it)
            # Sample replay buffer / batch
#             state_np, next_state_np, action, reward, done, mask = replay_buffer.sample(batch_size)
            state_np, next_state_np, action, reward = replay_buffer.sample(batch_size)
            state           = torch.FloatTensor(state_np).to(device)
            action          = torch.FloatTensor(action).to(device)
            next_state      = torch.FloatTensor(next_state_np).to(device)
            reward          = torch.FloatTensor(reward).to(device)
#             done            = torch.FloatTensor(1 - done).to(device)


            # Variational Auto-Encoder Training
            recon, mean, std = self.vae(state, action)
            recon_loss = F.mse_loss(recon, action)
            KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            vae_loss = recon_loss + 0.5 * KL_loss

            self.vae_optimizer.zero_grad()
            vae_loss.backward()
            self.vae_optimizer.step()

            # Critic Training
            with torch.no_grad():
                # Duplicate state 10 times
                state_rep = torch.FloatTensor(np.repeat(next_state_np, 10, axis=0)).to(device)
                
                # Compute value of perturbed actions sampled from the VAE
                if self.use_cloning:
                    target_Q1, target_Q2 = self.critic_target(state_rep, self.vae.decode(state_rep))
                else:
                    target_Q1, target_Q2 = self.critic_target(state_rep, self.actor_target(state_rep, self.vae.decode(state_rep)))

                # Soft Clipped Double Q-learning 
                target_Q = 0.75 * torch.min(target_Q1, target_Q2) + 0.25 * torch.max(target_Q1, target_Q2)
                target_Q = target_Q.view(batch_size, -1).max(1)[0].view(-1, 1)

#                 target_Q = reward + done * discount * target_Q
                target_Q = reward + discount * target_Q

            current_Q1, current_Q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()


            # Pertubation Model / Action Training
            sampled_actions = self.vae.decode(state)
            perturbed_actions = self.actor(state, sampled_actions)
            action_divergence = ((sampled_actions - perturbed_actions)**2).sum(-1)

            # Update through DPG
            actor_loss = -self.critic.q1(state, perturbed_actions).mean()
                
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()


            # Update Target Networks 
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                    
    def sample_A(self, states, to_numpy= True):

        return self.select_action(states)
    def get_A(self, states):
        return self.select_action(states)
    
    def get_A_prob(self,states):
        a = self.select_action(states)
        return np.eye(self.max_action+1)[a]
        
        
        
        
        
        
                    
            