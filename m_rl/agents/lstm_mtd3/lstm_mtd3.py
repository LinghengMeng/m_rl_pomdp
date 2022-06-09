import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from copy import deepcopy
import itertools

#######################################################################################

#######################################################################################
class MLPCritic(nn.Module):
    def __init__(self, obs_dim, act_dim,
                 mem_pre_lstm_hid_sizes=(128,),
                 mem_lstm_hid_sizes=(128,),
                 mem_after_lstm_hid_size=(128,),
                 cur_feature_hid_sizes=(128,),
                 post_comb_hid_sizes=(128,),
                 hist_with_past_act=False):
        super(MLPCritic, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hist_with_past_act = hist_with_past_act
        #
        self.mem_pre_lstm_layers = nn.ModuleList()
        self.mem_lstm_layers = nn.ModuleList()
        self.mem_after_lstm_layers = nn.ModuleList()

        self.cur_feature_layers = nn.ModuleList()
        self.post_combined_layers = nn.ModuleList()
        # Memory
        #    Pre-LSTM
        if self.hist_with_past_act:
            mem_pre_lstm_layer_size = [obs_dim + act_dim] + list(mem_pre_lstm_hid_sizes)
        else:
            mem_pre_lstm_layer_size = [obs_dim] + list(mem_pre_lstm_hid_sizes)
        for h in range(len(mem_pre_lstm_layer_size) - 1):
            self.mem_pre_lstm_layers += [nn.Linear(mem_pre_lstm_layer_size[h],
                                                   mem_pre_lstm_layer_size[h + 1]),
                                         nn.ReLU()]
        #    LSTM
        self.mem_lstm_layer_sizes = [mem_pre_lstm_layer_size[-1]] + list(mem_lstm_hid_sizes)
        for h in range(len(self.mem_lstm_layer_sizes) - 1):
            self.mem_lstm_layers += [
                nn.LSTM(self.mem_lstm_layer_sizes[h], self.mem_lstm_layer_sizes[h + 1], batch_first=True)]

        #   After-LSTM
        self.mem_after_lstm_layer_size = [self.mem_lstm_layer_sizes[-1]] + list(mem_after_lstm_hid_size)
        for h in range(len(self.mem_after_lstm_layer_size)-1):
            self.mem_after_lstm_layers += [nn.Linear(self.mem_after_lstm_layer_size[h],
                                                     self.mem_after_lstm_layer_size[h+1]),
                                           nn.ReLU()]

        # Current Feature Extraction
        cur_feature_layer_size = [obs_dim + act_dim] + list(cur_feature_hid_sizes)
        for h in range(len(cur_feature_layer_size) - 1):
            self.cur_feature_layers += [nn.Linear(cur_feature_layer_size[h], cur_feature_layer_size[h + 1]),
                                        nn.ReLU()]

        # Post-Combination
        post_combined_layer_size = [self.mem_after_lstm_layer_size[-1] + cur_feature_layer_size[-1]] + list(
            post_comb_hid_sizes) + [1]
        for h in range(len(post_combined_layer_size) - 2):
            self.post_combined_layers += [nn.Linear(post_combined_layer_size[h], post_combined_layer_size[h + 1]),
                                          nn.ReLU()]
        self.post_combined_layers += [nn.Linear(post_combined_layer_size[-2], post_combined_layer_size[-1]),
                                      nn.Identity()]

    def forward(self, obs, act, hist_obs, hist_act, hist_seg_len):
        #
        tmp_hist_seg_len = deepcopy(hist_seg_len)
        tmp_hist_seg_len[hist_seg_len == 0] = 1
        if self.hist_with_past_act:
            x = torch.cat([hist_obs, hist_act], dim=-1)
        else:
            x = hist_obs
        # Memory
        #    Pre-LSTM
        for layer in self.mem_pre_lstm_layers:
            x = layer(x)
        #    LSTM
        for layer in self.mem_lstm_layers:
            x, (lstm_hidden_state, lstm_cell_state) = layer(x)
        #    After-LSTM
        for layer in self.mem_after_lstm_layers:
            x = layer(x)
        #    History output mask to reduce disturbance cased by none history memory
        hist_out = torch.gather(x, 1,
                                (tmp_hist_seg_len - 1).view(-1, 1).repeat(1, self.mem_after_lstm_layer_size[-1]).unsqueeze(
                                    1).long()).squeeze(1)
        # Current Feature Extraction
        x = torch.cat([obs, act], dim=-1)
        for layer in self.cur_feature_layers:
            x = layer(x)

        extracted_memory = hist_out
        if len(extracted_memory) != len(x):
            import pdb; pdb.set_trace()
        x = torch.cat([extracted_memory, x], dim=-1)

        for layer in self.post_combined_layers:
            x = layer(x)
        # squeeze(x, -1) : critical to ensure q has right shape.
        return torch.squeeze(x, -1), hist_out, extracted_memory


class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit,
                 mem_pre_lstm_hid_sizes=(128,),
                 mem_lstm_hid_sizes=(128,),
                 mem_after_lstm_hid_size=(128,),
                 cur_feature_hid_sizes=(128,),
                 post_comb_hid_sizes=(128,),
                 hist_with_past_act=False):
        super(MLPActor, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.hist_with_past_act = hist_with_past_act
        #
        self.mem_pre_lstm_layers = nn.ModuleList()
        self.mem_lstm_layers = nn.ModuleList()
        self.mem_after_lstm_layers = nn.ModuleList()

        self.cur_feature_layers = nn.ModuleList()
        self.post_combined_layers = nn.ModuleList()

        # Memory
        #    Pre-LSTM
        if self.hist_with_past_act:
            mem_pre_lstm_layer_size = [obs_dim + act_dim] + list(mem_pre_lstm_hid_sizes)
        else:
            mem_pre_lstm_layer_size = [obs_dim] + list(mem_pre_lstm_hid_sizes)
        for h in range(len(mem_pre_lstm_layer_size) - 1):
            self.mem_pre_lstm_layers += [nn.Linear(mem_pre_lstm_layer_size[h],
                                                   mem_pre_lstm_layer_size[h + 1]),
                                         nn.ReLU()]
        #    LSTM
        self.mem_lstm_layer_sizes = [mem_pre_lstm_layer_size[-1]] + list(mem_lstm_hid_sizes)
        for h in range(len(self.mem_lstm_layer_sizes) - 1):
            self.mem_lstm_layers += [
                nn.LSTM(self.mem_lstm_layer_sizes[h], self.mem_lstm_layer_sizes[h + 1], batch_first=True)]
        #   After-LSTM
        self.mem_after_lstm_layer_size = [self.mem_lstm_layer_sizes[-1]] + list(mem_after_lstm_hid_size)
        for h in range(len(self.mem_after_lstm_layer_size) - 1):
            self.mem_after_lstm_layers += [nn.Linear(self.mem_after_lstm_layer_size[h],
                                                     self.mem_after_lstm_layer_size[h + 1]),
                                           nn.ReLU()]
        # Current Feature Extraction
        cur_feature_layer_size = [obs_dim] + list(cur_feature_hid_sizes)
        for h in range(len(cur_feature_layer_size) - 1):
            self.cur_feature_layers += [nn.Linear(cur_feature_layer_size[h], cur_feature_layer_size[h + 1]),
                                        nn.ReLU()]

        # Post-Combination
        post_combined_layer_size = [self.mem_after_lstm_layer_size[-1] + cur_feature_layer_size[-1]] + list(
            post_comb_hid_sizes) + [act_dim]
        for h in range(len(post_combined_layer_size) - 2):
            self.post_combined_layers += [nn.Linear(post_combined_layer_size[h], post_combined_layer_size[h + 1]),
                                          nn.ReLU()]
        self.post_combined_layers += [nn.Linear(post_combined_layer_size[-2], post_combined_layer_size[-1]), nn.Tanh()]

    def forward(self, obs, hist_obs, hist_act, hist_seg_len):
        #
        tmp_hist_seg_len = deepcopy(hist_seg_len)
        tmp_hist_seg_len[hist_seg_len == 0] = 1
        if self.hist_with_past_act:
            x = torch.cat([hist_obs, hist_act], dim=-1)
        else:
            x = hist_obs
        # Memory
        #    Pre-LSTM
        for layer in self.mem_pre_lstm_layers:
            x = layer(x)
        #    LSTM
        for layer in self.mem_lstm_layers:
            x, (lstm_hidden_state, lstm_cell_state) = layer(x)
        #    After-LSTM
        for layer in self.mem_after_lstm_layers:
            x = layer(x)
        hist_out = torch.gather(x, 1,
                                (tmp_hist_seg_len - 1).view(-1, 1).repeat(1, self.mem_after_lstm_layer_size[-1]).unsqueeze(
                                    1).long()).squeeze(1)
        # Current Feature Extraction
        x = obs
        for layer in self.cur_feature_layers:
            x = layer(x)

        # Post-Combination
        extracted_memory = hist_out
        x = torch.cat([extracted_memory, x], dim=-1)

        for layer in self.post_combined_layers:
            x = layer(x)
        return self.act_limit * x, hist_out, extracted_memory


class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit=1,
                 critic_mem_pre_lstm_hid_sizes=(128,),
                 critic_mem_lstm_hid_sizes=(128,),
                 critic_mem_after_lstm_hid_size=(128,),
                 critic_cur_feature_hid_sizes=(128,),
                 critic_post_comb_hid_sizes=(128,),
                 critic_hist_with_past_act=False,
                 actor_mem_pre_lstm_hid_sizes=(128,),
                 actor_mem_lstm_hid_sizes=(128,),
                 actor_mem_after_lstm_hid_size=(128,),
                 actor_cur_feature_hid_sizes=(128,),
                 actor_post_comb_hid_sizes=(128,),
                 actor_hist_with_past_act=False):
        super(MLPActorCritic, self).__init__()
        self.q1 = MLPCritic(obs_dim, act_dim,
                            mem_pre_lstm_hid_sizes=critic_mem_pre_lstm_hid_sizes,
                            mem_lstm_hid_sizes=critic_mem_lstm_hid_sizes,
                            mem_after_lstm_hid_size=critic_mem_after_lstm_hid_size,
                            cur_feature_hid_sizes=critic_cur_feature_hid_sizes,
                            post_comb_hid_sizes=critic_post_comb_hid_sizes,
                            hist_with_past_act=critic_hist_with_past_act)
        self.q2 = MLPCritic(obs_dim, act_dim,
                            mem_pre_lstm_hid_sizes=critic_mem_pre_lstm_hid_sizes,
                            mem_lstm_hid_sizes=critic_mem_lstm_hid_sizes,
                            mem_after_lstm_hid_size=critic_mem_after_lstm_hid_size,
                            cur_feature_hid_sizes=critic_cur_feature_hid_sizes,
                            post_comb_hid_sizes=critic_post_comb_hid_sizes,
                            hist_with_past_act=critic_hist_with_past_act)
        self.pi = MLPActor(obs_dim, act_dim, act_limit,
                           mem_pre_lstm_hid_sizes=actor_mem_pre_lstm_hid_sizes,
                           mem_lstm_hid_sizes=actor_mem_lstm_hid_sizes,
                           mem_after_lstm_hid_size=actor_mem_after_lstm_hid_size,
                           cur_feature_hid_sizes=actor_cur_feature_hid_sizes,
                           post_comb_hid_sizes=actor_post_comb_hid_sizes,
                           hist_with_past_act=actor_hist_with_past_act)

    def act(self, obs, hist_obs=None, hist_act=None, hist_seg_len=None, device=None):
        if (hist_obs is None) or (hist_act is None) or (hist_seg_len is None):
            hist_obs = torch.zeros(1, 1, self.obs_dim).to(device)
            hist_act = torch.zeros(1, 1, self.act_dim).to(device)
            hist_seg_len = torch.zeros(1).to(device)
        with torch.no_grad():
            act, _, _ = self.pi(obs, hist_obs, hist_act, hist_seg_len)
        return act.cpu().numpy()


class LSTMMTD3(object):
    """
    LSTM-Multistep TD3 includes both LSTM-based memory part and the multistep bootstrapping.
    """
    def __init__(self, obs_space, act_space, hidden_sizes,
                 multistep_size=5,
                 agent_mem_len=5,
                 gamma=0.99, polyak=0.995, pi_lr=1e-3, q_lr=1e-3,
                 start_steps=10000,
                 act_noise=0.1, target_noise=0.2, noise_clip=0.5,
                 update_after=1000, update_every=50, batch_size=64,
                 mem_manager=None, replay_size=1e6,
                 policy_delay=2, checkpoint_dir=None):
        #
        self.obs_space = obs_space
        self.act_space = act_space
        self.obs_dim = obs_space.shape[0]
        self.act_dim = act_space.shape[0]
        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_limit = act_space.high[0]
        self.hidden_sizes = hidden_sizes

        # Multistep size
        self.multistep_size = multistep_size

        self.act_noise = act_noise
        self.start_steps = start_steps

        self.gamma = gamma
        self.target_noise = target_noise
        self.noise_clip = noise_clip

        self.update_after = update_after
        self.update_every = update_every

        self.agent_mem_len = agent_mem_len

        # Create replay buffer
        self.obs = None
        self.act = None
        if self.agent_mem_len > 0:
            self.agent_mem_obs_buff = np.zeros([self.agent_mem_len, self.obs_dim])
            self.agent_mem_act_buff = np.zeros([self.agent_mem_len, self.act_dim])
            self.agent_mem_buff_len = 0
        else:
            self.agent_mem_obs_buff = np.zeros([1, self.obs_dim])
            self.agent_mem_act_buff = np.zeros([1, self.act_dim])
            self.agent_mem_buff_len = 0

        self.batch_size = batch_size
        self.mem_manager = mem_manager

        self.q_lr = q_lr
        self.pi_lr = pi_lr
        self.polyak = polyak
        self.policy_delay = policy_delay

        assert checkpoint_dir is not None, "Checkpoint_dir is None!"
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.cp_dir = checkpoint_dir

        # Initialize actor-critic
        self._init_actor_critic()

        # Gamma to the power of (step-1)
        self.gamma_power = torch.as_tensor([self.gamma ** i for i in range(self.multistep_size)], dtype=torch.float32).to(self.ac_device)
        self.gamma_power_batch = torch.tile(self.gamma_power, (self.batch_size, 1))

    def _init_actor_critic(self):

        # Create actor-critic module and target networks
        critic_mem_pre_lstm_hid_sizes = [128]
        critic_mem_lstm_hid_sizes = [128]
        critic_mem_after_lstm_hid_size = []
        critic_cur_feature_hid_sizes = [128, 128]
        critic_post_comb_hid_sizes = [128]
        critic_hist_with_past_act = True
        actor_mem_pre_lstm_hid_sizes = [128]
        actor_mem_lstm_hid_sizes = [128]
        actor_mem_after_lstm_hid_size = []
        actor_cur_feature_hid_sizes = [128, 128]
        actor_post_comb_hid_sizes = [128]
        actor_hist_with_past_act = True
        self.ac = MLPActorCritic(self.obs_dim, self.act_dim, self.act_limit,
                                 critic_mem_pre_lstm_hid_sizes=critic_mem_pre_lstm_hid_sizes,
                                 critic_mem_lstm_hid_sizes=critic_mem_lstm_hid_sizes,
                                 critic_mem_after_lstm_hid_size=critic_mem_after_lstm_hid_size,
                                 critic_cur_feature_hid_sizes=critic_cur_feature_hid_sizes,
                                 critic_post_comb_hid_sizes=critic_post_comb_hid_sizes,
                                 critic_hist_with_past_act=critic_hist_with_past_act,
                                 actor_mem_pre_lstm_hid_sizes=actor_mem_pre_lstm_hid_sizes,
                                 actor_mem_lstm_hid_sizes=actor_mem_lstm_hid_sizes,
                                 actor_mem_after_lstm_hid_size=actor_mem_after_lstm_hid_size,
                                 actor_cur_feature_hid_sizes=actor_cur_feature_hid_sizes,
                                 actor_post_comb_hid_sizes=actor_post_comb_hid_sizes,
                                 actor_hist_with_past_act=actor_hist_with_past_act)
        self.ac_targ = deepcopy(self.ac)

        self.ac_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ac.to(self.ac_device)
        self.ac_targ.to(self.ac_device)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.pi_lr)
        self.q_optimizer = Adam(self.q_params, lr=self.q_lr)

    def save_checkpoint(self):
        """Save learned reward network to disk."""
        save_elements = {'ac_state_dict': self.ac.state_dict(),
                         'ac_targ_state_dict': self.ac_targ.state_dict(),
                         'pi_optimizer_state_dict': self.pi_optimizer.state_dict(),
                         'q_optimizer_state_dict': self.q_optimizer.state_dict()}
        return save_elements

    def restore_checkpoint(self, restore_elements, mem_manager):
        self.ac.load_state_dict(restore_elements['ac_state_dict'])
        self.ac_targ.load_state_dict(restore_elements['ac_targ_state_dict'])
        self.pi_optimizer.load_state_dict(restore_elements['pi_optimizer_state_dict'])
        self.q_optimizer.load_state_dict(restore_elements['q_optimizer_state_dict'])
        print('Successfully restored Agent!')
        self.mem_manager = mem_manager

    def _compute_loss_q(self, data):
        """function for computing TD3 Q-losses"""
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        hc_r = data['hc_rew']
        h_o = data['agent_mem_seg_obs']
        h_a = data['agent_mem_seg_act']
        h_o2 = data['agent_mem_seg_obs2']
        h_a2 = data['agent_mem_seg_act2']
        h_len = data['agent_mem_seg_len']
        multistep_size = data['multistep_size']
        multistep_reward_seg = data['multistep_reward_seg']
        multistep_obs2 = data['multistep_obs2']
        multistep_done = data['multistep_done']

        q1, q1_hist_out, q1_extracted_memory = self.ac.q1(o, a, h_o, h_a, h_len)
        q2, q2_hist_out, q2_extracted_memory = self.ac.q2(o, a, h_o, h_a, h_len)

        # Bellman backup for Q functions
        with torch.no_grad():
            pi_targ, _, _ = self.ac_targ.pi(o2, h_o2, h_a2, h_len)

            # Target policy smoothing
            epsilon = torch.randn_like(pi_targ) * self.target_noise
            epsilon = torch.clamp(epsilon, -self.noise_clip, self.noise_clip)
            a2 = pi_targ + epsilon
            a2 = torch.clamp(a2, -self.act_limit, self.act_limit)

            # Target Q-values
            q1_pi_targ, _, _ = self.ac_targ.q1(o2, a2, h_o2, h_a2, h_len)
            q2_pi_targ, _, _ = self.ac_targ.q2(o2, a2, h_o2, h_a2, h_len)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)

            # Calculate multistep backup
            if self.gamma_power_batch.shape[0] != multistep_reward_seg.shape[0]:  # Avoid recalculate gamma_power if the bach size doesn't change.
                self.gamma_power_batch = torch.tile(self.gamma_power, (multistep_reward_seg.shape[0], 1))
            backup = torch.sum(multistep_reward_seg * self.gamma_power_batch, dim=1) + self.gamma ** multistep_size * (
                        1 - multistep_done) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        loss_info = dict(Q1Vals=q1.cpu().detach().numpy(),
                         Q2Vals=q2.cpu().detach().numpy())
        return loss_q, loss_info

    def _compute_loss_pi(self, data):
        """function for computing TD3 pi loss"""
        o = data['obs']
        h_o = data['agent_mem_seg_obs']
        h_a = data['agent_mem_seg_act']
        h_len = data['agent_mem_seg_len']
        a, _, _ = self.ac.pi(o, h_o, h_a, h_len)
        q1_pi, _, _ = self.ac.q1(o, a, h_o, h_a, h_len)
        return -q1_pi.mean()

    def get_train_action(self, obs, mem_obs_buff, mem_act_buff, mem_buff_len):
        o = torch.as_tensor(obs, dtype=torch.float32).view(1, -1).to(self.ac_device)
        mem_o = torch.tensor(mem_obs_buff).view(1, mem_obs_buff.shape[0], mem_obs_buff.shape[1]).float().to(self.ac_device)
        mem_a = torch.tensor(mem_act_buff).view(1, mem_act_buff.shape[0], mem_act_buff.shape[1]).float().to(self.ac_device)
        mem_l = torch.tensor([mem_buff_len]).float().to(self.ac_device)
        a = self.ac.act(o, mem_o, mem_a, mem_l).reshape(self.act_dim)
        a += self.act_noise * np.random.randn(self.act_dim)
        a = np.clip(a, -self.act_limit, self.act_limit)
        return a

    def get_test_action(self, obs, mem_obs_buff, mem_act_buff, mem_buff_len):
        o = torch.as_tensor(obs, dtype=torch.float32).view(1, -1).to(self.ac_device)
        mem_o = torch.tensor(mem_obs_buff).view(1, mem_obs_buff.shape[0], mem_obs_buff.shape[1]).float().to(
            self.ac_device)
        mem_a = torch.tensor(mem_act_buff).view(1, mem_act_buff.shape[0], mem_act_buff.shape[1]).float().to(
            self.ac_device)
        mem_l = torch.tensor([mem_buff_len]).float().to(self.ac_device)
        a = self.ac.act(o, mem_o, mem_a, mem_l).reshape(self.act_dim)
        return a

    def interact(self, time_step, new_obs, rew, hc_rew, done, info, terminal, logger):
        # If not the initial observation, store the latest experience (obs, act, rew, new_obs, done).
        if self.obs is not None:
            self.mem_manager.store_experience(self.obs, self.act, new_obs, rew, hc_rew, done, 'TD3_agent',
                                              obs_time=self.obs_timestamp, act_time=info['act_datetime'], obs2_time=info['obs_datetime'])

        # If terminal, start from the new episode where no previous (obs, act) exist.
        if terminal:
            self.obs = None
            self.act = None
            self.obs_timestamp = None
            if self.agent_mem_len > 0:
                self.agent_mem_obs_buff = np.zeros([self.agent_mem_len, self.obs_dim])
                self.agent_mem_act_buff = np.zeros([self.agent_mem_len, self.act_dim])
                self.agent_mem_buff_len = 0
            else:
                self.agent_mem_obs_buff = np.zeros([1, self.obs_dim])
                self.agent_mem_act_buff = np.zeros([1, self.act_dim])
                self.agent_mem_buff_len = 0
        else:
            # Get action:
            #   Until start_steps have elapsed, randomly sample actions
            #   from a uniform distribution for better exploration. Afterwards,
            #   use the learned policy (with some noise, via act_noise).
            if time_step > self.start_steps:
                self.act = self.get_train_action(new_obs, self.agent_mem_obs_buff,
                                                 self.agent_mem_act_buff, self.agent_mem_buff_len)
            else:
                self.act = self.act_space.sample()
            self.obs = new_obs
            self.obs_timestamp = info['obs_datetime']

            # Add short history
            if self.agent_mem_buff_len == self.agent_mem_len:
                self.agent_mem_obs_buff[:self.agent_mem_len - 1] = self.agent_mem_obs_buff[1:]
                self.agent_mem_act_buff[:self.agent_mem_len - 1] = self.agent_mem_act_buff[1:]
                self.agent_mem_obs_buff[self.agent_mem_len - 1] = list(self.obs)
                self.agent_mem_act_buff[self.agent_mem_len - 1] = list(self.act)
            else:
                self.agent_mem_obs_buff[self.agent_mem_buff_len + 1 - 1] = list(self.obs)
                self.agent_mem_act_buff[self.agent_mem_buff_len + 1 - 1] = list(self.act)
                self.agent_mem_buff_len += 1

        # Update
        logger = self.update(time_step, logger, done)

        return self.act, logger

    def update(self, time_step, logger, done=None):
        #
        if time_step >= self.update_after and time_step % self.update_every == 0:
            for j in range(self.update_every):
                # Sample batch from replay buffer and update agent
                batch = self.mem_manager.sample_exp_batch(self.batch_size, device=self.ac_device,
                                                          agent_mem_len=self.agent_mem_len,
                                                          multistep_size=self.multistep_size)

                # First run one gradient descent step for Q1 and Q2
                self.q_optimizer.zero_grad()
                loss_q, loss_info = self._compute_loss_q(batch)
                loss_q.backward()
                self.q_optimizer.step()

                # Record things
                logger.store(LossQ=loss_q.item(), **loss_info)

                # Possibly update pi and target networks
                if j % self.policy_delay == 0:
                    # Freeze Q-networks so you don't waste computational effort
                    # computing gradients for them during the policy learning step.
                    for p in self.q_params:
                        p.requires_grad = False

                    # Next run one gradient descent step for pi.
                    self.pi_optimizer.zero_grad()
                    loss_pi = self._compute_loss_pi(batch)
                    loss_pi.backward()
                    self.pi_optimizer.step()

                    # Unfreeze Q-networks so you can optimize it at next DDPG step.
                    for p in self.q_params:
                        p.requires_grad = True

                    # Record things
                    logger.store(LossPi=loss_pi.item())

                    # Finally, update target networks by polyak averaging.
                    with torch.no_grad():
                        for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                            # NB: We use an in-place operations "mul_", "add_" to update target
                            # params, as opposed to "mul" and "add", which would make new tensors.
                            p_targ.data.mul_(self.polyak)
                            p_targ.data.add_((1 - self.polyak) * p.data)
        return logger