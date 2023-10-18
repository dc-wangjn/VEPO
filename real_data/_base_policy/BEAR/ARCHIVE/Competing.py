        
##############################################################################################################################################################################################################################################################################################################################
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
            state_np, next_state_np, action, reward, done, mask = replay_buffer.sample(batch_size)
            state           = torch.FloatTensor(state_np).to(device)
            action          = torch.FloatTensor(action).to(device)
            next_state      = torch.FloatTensor(next_state_np).to(device)
            reward          = torch.FloatTensor(reward).to(device)
            done            = torch.FloatTensor(1 - done).to(device)


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

                target_Q = reward + done * discount * target_Q

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
            
        # DO ALL logging here
        logger.record_dict(create_stats_ordered_dict(
            'Q_target',
            target_Q.cpu().data.numpy(),
        ))
        logger.record_tabular('Actor Loss', actor_loss.cpu().data.numpy())
        logger.record_tabular('Critic Loss', critic_loss.cpu().data.numpy())
        # logger.record_tabular('Std Loss', std_loss.cpu().data.numpy().mean())
        logger.record_dict(create_stats_ordered_dict(
            'Sampled Actions',
            sampled_actions.cpu().data.numpy()
        ))
        logger.record_dict(create_stats_ordered_dict(
            'Perturbed Actions',
            perturbed_actions.cpu().data.numpy()
        ))
        logger.record_dict(create_stats_ordered_dict(
            'Current_Q',
            current_Q1.cpu().data.numpy()
        ))
        logger.record_dict(create_stats_ordered_dict(
            'Action_Divergence',
            action_divergence.cpu().data.numpy()
        ))
        
class Actor(nn.Module):
    """Actor used in BCQ"""
    def __init__(self, state_dim, action_dim, max_action, h_dim = 10, threshold=0.05):
        """ ??? """
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, h_dim)
        self.l2 = nn.Linear(h_dim, h_dim)
        self.l3 = nn.Linear(h_dim, action_dim)
        
        self.max_action = max_action
        self.threshold = threshold

    def forward(self, state, action):
        """ ??? """
        a = F.relu(self.l1(torch.cat([state, action], 1)))
        a = F.relu(self.l2(a))
        a = self.threshold * self.max_action * torch.tanh(self.l3(a))
        """ ??? """
        return (a + action).clamp(-self.max_action, self.max_action)

##########################################################################################################################################################################################################################################################################################################################################

class DQfD(object):
    """Deep Q-Learning from Demonstrations (but with only static data)"""
    def __init__(self, state_dim, action_dim, max_action, lambda_=0.4, margin_threshold=1.0):
        latent_dim = action_dim * 2

        self.actor = ActorTD3(state_dim, action_dim, max_action).to(device)
        self.actor_target = ActorTD3(state_dim, action_dim, max_action).to(device)
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
        self.lambda_ = lambda_
        # self.margin_fn = margin_fn
        self.margin_threshold = margin_threshold

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005):
        policy_noise = 0.2
        noise_clip = 0.5

        for it in range(iterations):
            state_np, next_state_np, action_np, reward, done, mask = replay_buffer.sample(batch_size)
            state           = torch.FloatTensor(state_np).to(device)
            action          = torch.FloatTensor(action_np).to(device)
            next_state      = torch.FloatTensor(next_state_np).to(device)
            reward          = torch.FloatTensor(reward).to(device)
            done            = torch.FloatTensor(1 - done).to(device)

            # Variational Auto-Encoder Training: fit the data collection policy
            recon, mean, std = self.vae(state, action)
            recon_loss = F.mse_loss(recon, action)
            KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            vae_loss = recon_loss + 0.5 * KL_loss
            # print ('VAE Loss: ', vae_loss.item())

            self.vae_optimizer.zero_grad()
            vae_loss.backward()
            self.vae_optimizer.step()

            noise = torch.FloatTensor(action_np).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)
            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            
            ## Additional imitation loss
            actor_actions, pre_tanh_actor_actions = self.actor.forward(state, preval=True)
            # actor_actions, raw_actor_actions = self.actor.sample_multiple(state, num_sample=1)
            sampled_actions, raw_sampled_actions = self.vae.decode_multiple(state, num_decode=1)
            # print ('Sampled Actions: ', sampled_actions.mean().item(), actor_actions.mean().item(), pre_tanh_actor_actions.mean().item())
            actor_Q = self.critic.q1(state, actor_actions[:, :])
            sampled_Q = self.critic.q1(state, sampled_actions[:, 0, :])
            # print ('Q-vals: ', actor_Q.mean().item(), sampled_Q.mean().item())

            action_divergence = ((sampled_actions[:, 0, :] - actor_actions)**2).sum(-1)
            # import ipdb; ipdb.set_trace()
            margin_loss = torch.nn.ReLU()(actor_Q + self.margin_threshold - sampled_Q).mean()
            # margin_loss = (torch.max(actor_Q + self.margin_threshold, sampled_Q) - sampled_Q + 1e-6).mean()
            
            # Adding these losses
            critic_loss = critic_loss + self.lambda_ * margin_loss
            # print ('Critic/MArgin Loss : ', critic_loss.item(), margin_loss.item())
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            if it % 2 == 0:
                actor_actions = self.actor(state)
                # actor_actions, _ = self.actor.sample_multiple(state, num_sample=1)
                actor_loss = -self.critic.q1(state, actor_actions[:, :]).mean()
                # print ('Actor Loss: ', actor_loss.item())
                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        logger.record_dict(create_stats_ordered_dict(
            'Q_target',
            target_Q.cpu().data.numpy(),
        ))
        logger.record_tabular('Actor Loss', actor_loss.cpu().data.numpy())
        logger.record_tabular('Critic Loss', critic_loss.cpu().data.numpy())
        logger.record_tabular('Margin Loss', margin_loss.cpu().data.numpy())

        logger.record_dict(create_stats_ordered_dict(
            'Sampled Actions',
            sampled_actions.cpu().data.numpy()
        ))
        logger.record_dict(create_stats_ordered_dict(
            'Action_Divergence',
            action_divergence.cpu().data.numpy()
        ))
        logger.record_dict(create_stats_ordered_dict(
            'Current_Q',
            current_Q1.cpu().data.numpy()
        ))
        

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))


    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
        
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

##########################################################################################################################################################################################################################################################################################################################################

class KLControl(object):
    def __init__(self, num_qs, state_dim, action_dim, max_action, use_bootstrap=False, temperature=0.1, mode='auto'):
        latent_dim = action_dim * 2
        self.actor = RegularActor(state_dim, action_dim, max_action).to(device)
        self.actor_target = RegularActor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.vae = RegularActor(state_dim, action_dim, max_action).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters())

        self.max_action = max_action
        self.action_dim = action_dim
        self.use_bootstrap = use_bootstrap
        self.num_qs = num_qs
        self.temperature = temperature
        self.mode = mode

        if self.mode == 'auto':
            self.temperature = torch.randn( (), requires_grad=True, device=device)
            self.lagrange_opt = torch.optim.Adam([self.temperature,], lr=1e-3)
            self.kl_div_thresh = temperature

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.98, tau=0.005):
        for it in range(iterations):
            state_np, next_state_np, action_np, reward, done, mask = replay_buffer.sample(batch_size)
            state           = torch.FloatTensor(state_np).to(device)
            action          = torch.FloatTensor(action_np).to(device)
            next_state      = torch.FloatTensor(next_state_np).to(device)
            reward          = torch.FloatTensor(reward).to(device)
            done            = torch.FloatTensor(1 - done).to(device)

            # Variational Auto-Encoder Training: fit the data collection policy
            # recon, mean, std = self.vae(state, action)
            # recon_loss = F.mse_loss(recon, action)
            # KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            # vae_loss = recon_loss + 0.5 * KL_loss
            # print ('VAE Loss: ', vae_loss.item())
            # import ipdb; ipdb.set_trace()
            vae_log_pis = self.vae.log_pis(state=state, action=action)
            vae_loss = (-vae_log_pis).mean()
            # print ('VAE Loss: ', vae_loss.item())

            self.vae_optimizer.zero_grad()
            vae_loss.backward()
            self.vae_optimizer.step()

            next_action, raw_action = self.actor.sample_multiple(next_state, num_sample=1)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action[:, 0, :])
            kl_correction = -self.actor.log_pis(state=next_state, raw_action=raw_action[:, 0, :]) +\
                                    self.vae.log_pis(state=next_state, raw_action=raw_action[:, 0, :])
            if self.mode == 'auto':
                    target_Q = torch.min(target_Q1, target_Q2) + self.temperature.exp() * kl_correction.unsqueeze(-1)
            else:
                    target_Q = torch.min(target_Q1, target_Q2) + self.temperature * kl_correction.unsqueeze(-1)
            target_Q = reward + (done * discount * target_Q).detach()

            current_Q1, current_Q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            # print ('Critic Loss: ', critic_loss.item())

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            if it % 2 == 0:
                actor_actions, raw_actor_actions = self.actor.sample_multiple(state, num_sample=1)
                actor_log_pis = self.actor.log_pis(state=state, raw_action=raw_actor_actions[:, 0, :])
                vae_log_pis = self.vae.log_pis(state=state, raw_action=raw_actor_actions[:, 0, :])
                if self.mode == 'auto':
                        actor_loss = -self.critic.q1(state, actor_actions[:, 0, :]).mean() + self.temperature.exp() * (actor_log_pis - vae_log_pis).mean()
                else:
                        actor_loss = -self.critic.q1(state, actor_actions[:, 0, :]).mean() + self.temperature * (actor_log_pis - vae_log_pis).mean()
                estimated_kl = (actor_log_pis - vae_log_pis)
                # print ('Actor Loss: ', actor_loss.item(), ' Estimated KL: ', estimated_kl.mean().item())

                # Now the backprop through the temperature
                if self.mode == 'auto':
                        temperature_loss = (vae_log_pis - actor_log_pis + self.kl_div_thresh).mean() * self.temperature.exp()
                        self.lagrange_opt.zero_grad()
                        temperature_loss.backward(retain_graph=True)
                        self.lagrange_opt.step()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        logger.record_dict(create_stats_ordered_dict(
                'Q_target',
                target_Q.cpu().data.numpy(),
        ))
        logger.record_tabular('Actor Loss', actor_loss.cpu().data.numpy())
        logger.record_tabular('Critic Loss', critic_loss.cpu().data.numpy())
        logger.record_tabular('VAE Loss', vae_loss.cpu().data.numpy())
        if self.mode == 'auto':
                logger.record_tabular('Temperature Loss', temperature_loss.cpu().data.numpy())
                logger.record_tabular('Temperature', self.temperature.exp().cpu().data.numpy())
        logger.record_dict(create_stats_ordered_dict(
                'Current_Q',
                current_Q1.cpu().data.numpy()
        ))
        logger.record_dict(create_stats_ordered_dict(
                'Estimated KL',
                estimated_kl.cpu().data.numpy(),
        ))
