import numpy as np
import torch
from torch.optim import Adam
import gym
# import pybulletgym
import pybullet_envs
import time
import m_rl.agents.ppo.core as core
from m_rl.utils.logx import EpochLogger, setup_logger_kwargs
from m_rl.utils.tools import statistics_scalar
import os.path as osp
import os


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95, adv_use_gae_lambda=True, multistep_size=5, v_use_multistep_return=True):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.adv_use_gae_lambda = adv_use_gae_lambda
        self.multistep_size = multistep_size
        self.v_use_multistep_return = v_use_multistep_return
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        if self.adv_use_gae_lambda:
            # the next two lines implement GAE-Lambda advantage calculation
            deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
            self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

            # the next line computes rewards-to-go, to be targets for the value function
            self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        else:
            path_len = len(rews[:-1])
            if path_len < self.multistep_size:
                tmp_multistep_size = path_len
            else:
                tmp_multistep_size = self.multistep_size
            stacked_rews = np.zeros((tmp_multistep_size + 1, path_len))
            stacked_gamma = np.ones((tmp_multistep_size + 1, path_len))
            for s_i in range(tmp_multistep_size + 1):
                stacked_rews[s_i, :(path_len - s_i)] = rews[s_i:-1]
                stacked_gamma[s_i, :(path_len - s_i)] = self.gamma ** s_i
            stacked_rews[tmp_multistep_size, :] = np.asarray(vals[tmp_multistep_size:].tolist() + [vals[-1] for s_i in range(tmp_multistep_size - 1)])
            stacked_gamma[tmp_multistep_size, -tmp_multistep_size:] = self.gamma
            multistep_return = np.sum(stacked_rews * stacked_gamma, axis=0)
            self.adv_buf[path_slice] = multistep_return - vals[:-1]
            if self.v_use_multistep_return:
                self.ret_buf[path_slice] = multistep_return
            else:
                # the next line computes rewards-to-go, to be targets for the value function
                self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self, device=None):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k,v in data.items()}


class PPO(object):
    def __init__(self, obs_space, act_space, hidden_sizes,
                 gamma=0.99, clip_ratio=0.2,
                 adv_use_gae_lambda=True, multistep_size=5, v_use_multistep_return=True,
                 steps_per_epoch=4000,
                 train_pi_iters=80, train_v_iters = 80, target_kl = 0.01,
                 pi_lr=3e-4, vf_lr=1e-3, mem_manager=None, checkpoint_dir=None):
        self.obs_space = obs_space
        self.act_space = act_space
        self.hidden_sizes = hidden_sizes

        self.gamma = gamma
        self.clip_ratio = clip_ratio

        self.adv_use_gae_lambda = adv_use_gae_lambda  # Use GAE-Lambda advantage. Otherwise, use multistep advantage estimator
        self.multistep_size = multistep_size  # When use multistep advantage estimator, the multistep size
        self.v_use_multistep_return = v_use_multistep_return  # When use multistep advantage estimator, the way V-value estimation

        # Set up experience buffer
        self.steps_per_epoch = steps_per_epoch
        obs_dim = obs_space.shape[0]
        act_dim = act_space.shape[0]
        lam = 0.97    # lam (float): Lambda for GAE-Lambda. (Always between 0 and 1, close to 1.)
        self.buf = PPOBuffer(obs_dim, act_dim, self.steps_per_epoch, gamma, lam)

        self.mem_manager = mem_manager

        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.target_kl = target_kl

        # Variables to store experiences
        self.obs = None
        self.act = None
        self.v = None
        self.logp = None

        assert checkpoint_dir is not None, "Checkpoint_dir is None!"
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.cp_dir = checkpoint_dir

        # Initialize actor-critic
        self._init_actor_critic()

    def _init_actor_critic(self):
        # Set up function for computing PPO policy loss
        # Create actor-critic module
        self.ac = core.MLPActorCritic(self.obs_space, self.act_space, self.hidden_sizes)
        self.ac_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ac.to(self.ac_device)
        # Set up optimizers for policy and value function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.pi_lr)
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=self.vf_lr)

    def _compute_loss_pi(self, data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = self.ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def _compute_loss_v(self, data):
        obs, ret = data['obs'], data['ret']
        return ((self.ac.v(obs) - ret) ** 2).mean()

    def get_train_action(self, obs):
        a, v, logp = self.ac.step(torch.as_tensor(obs, dtype=torch.float32).to(self.ac_device))
        return a, v, logp

    # Note: PPO is on-policy RL, so don't need get_test_action.
    def get_test_action(self, obs):
        a, v, logp = self.ac.step(torch.as_tensor(obs, dtype=torch.float32).to(self.ac_device))
        return a

    def interact(self, time_step, new_obs, rew, hc_rew, done, info, terminal, logger):
        # If not the initial observation, store the latest experience (obs, act, rew, new_obs, done).
        if self.obs is not None:
            self.mem_manager.store_experience(self.obs, self.act, new_obs, rew, hc_rew, done, 'PPO_agent',
                                              obs_time=self.obs_timestamp, act_time=info['act_datetime'], obs2_time=info['obs_datetime'])
            self.buf.store(self.obs, self.act, rew, self.v, self.logp)
            logger.store(VVals=self.v)
            # print('time_step={}, ptr={}'.format(time_step, self.buf.ptr))

        # Select action
        if terminal:
            # if trajectory didn't reach done state, bootstrap value target
            if done:
                v = 0
            else:
                _, v, _ = self.get_train_action(torch.as_tensor(new_obs, dtype=torch.float32).to(self.ac_device))
            self.buf.finish_path(v)

            self.obs = None
            self.act = None
            self.obs_timestamp = None
            self.v = None
            self.logp = None
        else:
            self.act, self.v, self.logp = self.ac.step(torch.as_tensor(new_obs, dtype=torch.float32).to(self.ac_device))
            self.obs = new_obs
            self.obs_timestamp = info['obs_datetime']

        if self.buf.ptr == self.steps_per_epoch:
            logger = self.update(logger)

        return self.act, logger

    def update(self, logger):
        data = self.buf.get(self.ac_device)

        pi_l_old, pi_info_old = self._compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = self._compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self._compute_loss_pi(data)
            kl = pi_info['kl']
            if kl > 1.5 * self.target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.' % i)
                break
            loss_pi.backward()
            self.pi_optimizer.step()

        logger.store(StopIter=i)

        # Value function learning
        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self._compute_loss_v(data)
            loss_v.backward()
            self.vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))
        return logger

    def save_checkpoint(self):
        save_elements = {
            'buffer': self.buf,
            'ac_state_dict': self.ac.state_dict(),
            'pi_optimizer_state_dict': self.pi_optimizer.state_dict(),
            'vf_optimizer_state_dict': self.vf_optimizer.state_dict()}
        return save_elements

    def restore_checkpoint(self, restore_elements, mem_manager):
        self.buf = restore_elements['buffer']
        self.ac.load_state_dict(restore_elements['ac_state_dict'])
        self.pi_optimizer.load_state_dict(restore_elements['pi_optimizer_state_dict'])
        self.vf_optimizer.load_state_dict(restore_elements['vf_optimizer_state_dict'])
        print('Successfully restored Agent!')
        self.mem_manager = mem_manager