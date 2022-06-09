import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from copy import deepcopy
import itertools


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for agents.
    """

    def __init__(self, obs_dim, act_dim, replay_size=1e6):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.replay_size = int(replay_size)
        self.obs_buf = np.zeros((self.replay_size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros((self.replay_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((self.replay_size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(self.replay_size, dtype=np.float32)
        self.done_buf = np.zeros(self.replay_size, dtype=np.float32)
        self.ptr, self.size = 0, 0

    def store(self, obs, act, next_obs, rew, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.replay_size
        self.size = min(self.size+1, self.replay_size)

    # def sample_batch(self, batch_size=32):
    #     idxs = np.random.randint(0, self.size, size=batch_size)
    #     batch = dict(obs=self.obs_buf[idxs],
    #                  obs2=self.obs2_buf[idxs],
    #                  act=self.act_buf[idxs],
    #                  rew=self.rew_buf[idxs],
    #                  done=self.done_buf[idxs])
    #     return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

    def sample_batch(self, batch_size=32, device=None, agent_mem_len=None, reward_mem_len=None):

        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        if reward_mem_len is not None:
            # Extract reward memory
            batch['rew_mem_seg_len'] = np.zeros(batch_size)
            batch['rew_mem_seg_obs'] = np.zeros((batch_size, reward_mem_len, self.obs_dim))
            batch['rew_mem_seg_act'] = np.zeros((batch_size, reward_mem_len, self.act_dim))
            batch['rew_mem_seg_obs2'] = np.zeros((batch_size, reward_mem_len, self.obs_dim))

        if agent_mem_len is not None:
            # Extract agent memory
            batch['agent_mem_seg_len'] = np.zeros(batch_size)
            batch['agent_mem_seg_obs'] = np.zeros((batch_size, agent_mem_len, self.obs_dim))
            batch['agent_mem_seg_act'] = np.zeros((batch_size, agent_mem_len, self.act_dim))
            batch['agent_mem_seg_obs2'] = np.zeros((batch_size, agent_mem_len, self.obs_dim))
            batch['agent_mem_seg_act2'] = np.zeros((batch_size, agent_mem_len, self.act_dim))

        for i, id in enumerate(idxs):
            # Extract reward memory
            if reward_mem_len is not None:
                reward_mem_start_id = id - reward_mem_len + 1
                if reward_mem_start_id < 0:
                    reward_mem_start_id = 0
                # If exist done before the last experience (not include the done in id), start from the index next to the done.
                if len(np.where(self.done_buf[reward_mem_start_id:id]==1)[0]) != 0:
                    reward_mem_start_id = reward_mem_start_id + (np.where(self.done_buf[reward_mem_start_id:id] == 1)[0][-1])+1
                reward_mem_seg_len = id - reward_mem_start_id + 1
                batch['rew_mem_seg_len'][i] = reward_mem_seg_len
                batch['rew_mem_seg_obs'][i, :reward_mem_seg_len, :] = self.obs_buf[reward_mem_start_id:id+1]
                batch['rew_mem_seg_act'][i, :reward_mem_seg_len, :] = self.act_buf[reward_mem_start_id:id+1]
                batch['rew_mem_seg_obs2'][i, :reward_mem_seg_len, :] = self.obs2_buf[reward_mem_start_id:id + 1]

            # Extract agent memory
            # (Note agent memory does not include the current obs and act, but only includes (obs, act) pairs prior to
            #   the current (obs, act).)
            if agent_mem_len is not None:
                agent_mem_start_id = id - agent_mem_len
                if agent_mem_start_id < 0:
                    agent_mem_start_id = 0
                # If exist done before the last experience (not include the done in id), start from the index next to the done.
                if len(np.where(self.done_buf[agent_mem_start_id:id] == 1)[0]) != 0:
                    agent_mem_start_id = agent_mem_start_id + (
                    np.where(self.done_buf[agent_mem_start_id:id] == 1)[0][-1]) + 1
                agent_mem_seg_len = id - agent_mem_start_id
                batch['agent_mem_seg_len'][i] = agent_mem_seg_len
                batch['agent_mem_seg_obs'][i, :agent_mem_seg_len, :] = self.obs_buf[agent_mem_start_id:id]
                batch['agent_mem_seg_act'][i, :agent_mem_seg_len, :] = self.act_buf[agent_mem_start_id:id]
                batch['agent_mem_seg_obs2'][i, :agent_mem_seg_len, :] = self.obs2_buf[agent_mem_start_id:id]
                batch['agent_mem_seg_act2'][i, :agent_mem_seg_len, :] = self.act_buf[agent_mem_start_id+1:id+1]
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k, v in batch.items()}


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


class LSTMTD3(object):
    def __init__(self, obs_space, act_space, hidden_sizes,
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

            backup = r + self.gamma * (1 - d) * q_pi_targ

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
                batch = self.mem_manager.sample_exp_batch(self.batch_size, device=self.ac_device, agent_mem_len=self.agent_mem_len)

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