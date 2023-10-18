
class BEAR_IS(object):
    """Using Importance sampling on the bellman error, essentially making it agnostic of the 
       behaviour policy density, and care only about the support"""
    def __init__(self, num_qs, state_dim, action_dim, num_A, delta_conf=0.1, use_bootstrap=True, version=0, lambda_=0.4, threshold=0.05,
                 mode='hardcoded', num_samples_match=10, mmd_sigma=10.0, lagrange_thresh=10.0, use_kl=False, use_ensemble=True, kernel_type='laplacian'):
        latent_dim = action_dim * 2
        self.actor = RegularActor(state_dim, action_dim, num_A).to(device)
        self.actor_target = RegularActor(state_dim, action_dim, num_A).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = EnsembleCritic(num_qs, state_dim, action_dim).to(device)
        self.critic_target = EnsembleCritic(num_qs, state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.vae = VAE(state_dim, action_dim, latent_dim, num_A).to(device)
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
        
        if self.mode == 'auto':
            self.log_lagrange2 = torch.randn((), requires_grad=True, device=device)
            self.lagrange2_opt = torch.optim.Adam([self.log_lagrange2,], lr=1e-3)

        self.epoch = 0

    def mmd_loss_laplacian(self, samples1, samples2, sigma=0.2):
        """MMD Loss with Laplacian kernel for matching supports"""
        diff_x_x = samples1.unsqueeze(2) - samples1.unsqueeze(1)  # B x N x N x d
        diff_x_x = torch.mean((-(diff_x_x.abs()).sum(-1)/(2.0 * sigma)).exp(), dim=(1,2))

        diff_x_y = samples1.unsqueeze(2) - samples2.unsqueeze(1)
        diff_x_y = torch.mean((-(diff_x_y.abs()).sum(-1)/(2.0 * sigma)).exp(), dim=(1, 2))

        diff_y_y = samples2.unsqueeze(2) - samples2.unsqueeze(1)  # B x N x N x d
        diff_y_y = torch.mean((-(diff_y_y.abs()).sum(-1)/(2.0 * sigma)).exp(), dim=(1,2))

        overall_loss = (diff_x_x + diff_y_y - 2.0 * diff_x_y + 1e-6).sqrt()
        return overall_loss

    def mmd_loss_gaussian(self, samples1, samples2, sigma=0.2):
        """MMD constraint with Gaussian Kernel for support matching"""
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
        # import ipdb; ipdb.set_trace()
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
        # print (samples1_log_prob.min(), samples1_log_prob.max())
        samples1_prob = samples1_log_prob.clamp(min=-5, max=4).exp()
        return (samples1_prob).mean(1)
    
    def select_action(self, state):      
        # TODO (aviralkumar): Check this out once!  
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).repeat(10, 1).to(device)
            action = self.actor(state)
            q1 = self.critic.q1(state, action)
            ind = q1.max(0)[1]
        return action[ind].cpu().data.numpy().flatten()
    
    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005):
        for it in range(iterations):
            # Sample replay buffer / batch
            state_np, next_state_np, action, reward, done, mask, data_mean, data_cov = replay_buffer.sample(batch_size, with_data_policy=True)
            state           = torch.FloatTensor(state_np).to(device)
            action          = torch.FloatTensor(action).to(device)
            next_state      = torch.FloatTensor(next_state_np).to(device)
            reward          = torch.FloatTensor(reward).to(device)
            done            = torch.FloatTensor(1 - done).to(device)
            mask            = torch.FloatTensor(mask).to(device)
            data_mean       = torch.FloatTensor(data_mean).to(device)
            data_cov        = torch.FloatTensor(data_cov).to(device)

            # Variational Auto-Encoder Training: fit the data collection policy
            recon, mean, std = self.vae(state, action)
            recon_loss = F.mse_loss(recon, action)
            KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            vae_loss = recon_loss + 0.5 * KL_loss

            self.vae_optimizer.zero_grad()
            vae_loss.backward()
            self.vae_optimizer.step()

            # compute policy probs
            raw_action = atanh(action)
            policy_dist = td.Normal(data_mean, data_cov.exp())
            log_raw_action = policy_dist.log_prob(raw_action)
            log_raw_action = log_raw_action.sum(-1)
            tanh_correction = torch.log((1 - action**2 + 1e-6)).sum(-1)
            log_action_prob = log_raw_action - tanh_correction    # [B]

            # Critic Training: In this step, we explicitly compute the actions 
            with torch.no_grad():
                # Duplicate state 10 times
                state_rep = torch.FloatTensor(np.repeat(next_state_np, 10, axis=0)).to(device)
                
                # Compute value of perturbed actions sampled from the VAE
                target_Qs = self.critic_target(state_rep, self.actor_target(state_rep))

                # Soft-convex combination for target values
                target_Q = 0.75 * target_Qs.min(0)[0] + 0.25 * target_Qs.max(0)[0]
                target_Q = target_Q.view(batch_size, -1).max(1)[0].view(-1, 1)
                target_Q = reward + done * discount * target_Q

            current_Qs = self.critic(state, action, with_var=False)
            if self.use_bootstrap: 
                critic_loss = (F.mse_loss(current_Qs[0], target_Q, reduction='none') * mask[:, 0:1]).mean() +\
                            (F.mse_loss(current_Qs[1], target_Q, reduction='none') * mask[:, 1:2]).mean() +\
                            (F.mse_loss(current_Qs[2], target_Q, reduction='none') * mask[:, 2:3]).mean() +\
                            (F.mse_loss(current_Qs[3], target_Q, reduction='none') * mask[:, 3:4]).mean()
            else:
                critic_loss = weighted_mse_loss(current_Qs[0], target_Q, log_raw_action.exp()) +\
                         weighted_mse_loss(current_Qs[1], target_Q, log_raw_action.exp()) +\
                         weighted_mse_loss(current_Qs[2], target_Q, log_raw_action.exp()) +\
                         weighted_mse_loss(current_Qs[3], target_Q, log_raw_action.exp())

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Action Training
            # If you take less samples, it is closer to a uniform support set matching
            num_samples = self.num_samples_match
            sampled_actions, raw_sampled_actions = self.vae.decode_multiple(state, num_decode=num_samples)  # B x N x d
            actor_actions, raw_actor_actions = self.actor.sample_multiple(state, num_samples)#  num)
            if self.use_kl:
                mmd_loss = self.kl_loss(raw_sampled_actions, state)
            else:
                if self.kernel_type == 'gaussian':
                    mmd_loss = self.mmd_loss_gaussian(raw_sampled_actions, raw_actor_actions, sigma=self.mmd_sigma)
                else:
                    mmd_loss = self.mmd_loss_laplacian(raw_sampled_actions, raw_actor_actions, sigma=self.mmd_sigma)

            action_divergence = ((sampled_actions - actor_actions)**2).sum(-1)
            raw_action_divergence = ((raw_sampled_actions - raw_actor_actions)**2).sum(-1)

            # if self.use_kl or True:
            #     ent_loss = self.entropy_loss(raw_actor_actions, state)
            #     if self.epoch >= 50:
            #         mmd_loss = mmd_loss + 0.0001*ent_loss

            # Update through DPG
            critic_qs, std_q = self.critic.q_all(state, actor_actions[:, 0, :], with_var=True)
            critic_qs = self.critic.q_all(state.unsqueeze(0).repeat(num_samples, 1, 1).view(num_samples*state.size(0), state.size(1)), actor_actions.permute(1, 0, 2).contiguous().view(num_samples*actor_actions.size(0), actor_actions.size(2)))
            critic_qs = critic_qs.view(self.num_qs, num_samples, actor_actions.size(0), 1)
            critic_qs = critic_qs.mean(1)
            std_q = torch.std(critic_qs, dim=0, keepdim=False, unbiased=False)

            if not self.use_ensemble:
                std_q = torch.zeros_like(std_q).to(device)
                
            if self.version == '0':
                critic_qs = critic_qs.min(0)[0]
            elif self.version == '1':
                critic_qs = critic_qs.max(0)[0]
            elif self.version == '2':
                critic_qs = critic_qs.mean(0)
            # import ipdb; ipdb.set_trace()

            if self.epoch >= 20: 
                if self.mode == 'auto':
                    actor_loss = (-critic_qs +\
                        self._lambda * (np.sqrt((1 - self.delta_conf)/self.delta_conf)) * std_q +\
                        self.log_lagrange2.exp() * mmd_loss).mean()
                else:
                    actor_loss = (-critic_qs +\
                        self._lambda * (np.sqrt((1 - self.delta_conf)/self.delta_conf)) * std_q +\
                        100.0*mmd_loss).mean()
            else:
                if self.mode == 'auto':
                    actor_loss = (self.log_lagrange2.exp() * mmd_loss).mean()
                else:
                    actor_loss = 100.0*mmd_loss.mean()

            std_loss = self._lambda*(np.sqrt((1 - self.delta_conf)/self.delta_conf)) * std_q.detach() 

            self.actor_optimizer.zero_grad()
            if self.mode =='auto':
                actor_loss.backward(retain_graph=True)
            else:
                actor_loss.backward()
            self.actor_optimizer.step()

            thresh = 0.05
            if self.use_kl:
                thresh = -2.0

            if self.mode == 'auto':
                lagrange_loss = (-critic_qs +\
                        self._lambda * (np.sqrt((1 - self.delta_conf)/self.delta_conf)) * (std_q) +\
                        self.log_lagrange2.exp() * (mmd_loss - thresh)).mean()

                self.lagrange2_opt.zero_grad()
                (-lagrange_loss).backward()
                # self.lagrange1_opt.step()
                self.lagrange2_opt.step() 
                self.log_lagrange2.data.clamp_(min=-5.0, max=self.lagrange_thresh)   

            # Update Target Networks 
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # Do all logging here
        logger.record_dict(create_stats_ordered_dict(
            'Q_target',
            target_Q.cpu().data.numpy(),
        ))
        if self.mode == 'auto':
            if self.use_function_approx == 'False':
                logger.record_tabular('Lagrange2', self.log_lagrange2.exp().cpu().data.numpy())
            else:
                logger.record_dict(create_stats_ordered_dict(
                    'Lagrange2',
                    self.log_lagrange2(state).exp().cpu().data.numpy()
                ))

        logger.record_tabular('Actor Loss', actor_loss.cpu().data.numpy())
        logger.record_tabular('Critic Loss', critic_loss.cpu().data.numpy())
        logger.record_tabular('Std Loss', std_loss.cpu().data.numpy().mean())
        # logger.record_tabular('MMD Loss', mmd_loss.cpu().data.numpy().mean())
        logger.record_dict(create_stats_ordered_dict(
            'MMD Loss',
            mmd_loss.cpu().data.numpy()
        ))
        if self.use_kl:
            logger.record_dict(create_stats_ordered_dict(
                'Ent Loss',
                ent_loss.cpu().data.numpy()
            ))
        logger.record_dict(create_stats_ordered_dict(
            'Sampled Actions',
            sampled_actions.cpu().data.numpy()
        ))
        logger.record_dict(create_stats_ordered_dict(
            'Actor Actions',
            actor_actions.cpu().data.numpy()
        ))
        logger.record_dict(create_stats_ordered_dict(
            'Current_Q',
            current_Qs.cpu().data.numpy()
        ))
        logger.record_dict(create_stats_ordered_dict(
            'Action_Divergence',
            action_divergence.cpu().data.numpy()
        ))
        logger.record_dict(create_stats_ordered_dict(
            'Raw Action_Divergence',
            raw_action_divergence.cpu().data.numpy()
        ))
        self.epoch = self.epoch + 1

