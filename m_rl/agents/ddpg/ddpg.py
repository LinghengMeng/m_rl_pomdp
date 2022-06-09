import os
from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
from m_rl.agents.ddpg.core import MLPActorCritic, combined_shape


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for TD3 agents.
    """

    def __init__(self, obs_dim, act_dim, max_replay_size=1e6):
        # Experience replay buffer with longer history
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        max_replay_size = int(max_replay_size)
        self.obs_buf = np.zeros(combined_shape(max_replay_size, self.obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(max_replay_size, self.act_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(max_replay_size, self.obs_dim), dtype=np.float32)
        self.rew_buf = np.zeros(max_replay_size, dtype=np.float32)
        self.done_buf = np.zeros(max_replay_size, dtype=np.float32)
        self.ptr, self.size, self.replay_size = 0, 0, max_replay_size

    def store(self, obs, act, next_obs, rew, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.replay_size
        self.size = min(self.size+1, self.replay_size)

    # def sample_batch(self, batch_size=32, device=None):
    #     idxs = np.random.randint(0, self.size, size=batch_size)
    #     batch = dict(obs=self.obs_buf[idxs],
    #                  obs2=self.obs2_buf[idxs],
    #                  act=self.act_buf[idxs],
    #                  rew=self.rew_buf[idxs],
    #                  done=self.done_buf[idxs])
    #     return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k, v in batch.items()}

    def sample_batch(self, batch_size=32, device=None, mem_len=None):
        idxs = np.random.randint(0, self.size, size=batch_size)
        if mem_len is None:
            batch = dict(obs=self.obs_buf[idxs],
                         obs2=self.obs2_buf[idxs],
                         act=self.act_buf[idxs],
                         rew=self.rew_buf[idxs],
                         done=self.done_buf[idxs])
        else:
            batch = dict(obs=self.obs_buf[idxs],
                         obs2=self.obs2_buf[idxs],
                         act=self.act_buf[idxs],
                         rew=self.rew_buf[idxs],
                         done=self.done_buf[idxs])
            # Extract memory
            batch['mem_seg_len'] = np.zeros(batch_size)
            batch['mem_seg_obs'] = np.zeros((batch_size, mem_len, self.obs_dim))
            batch['mem_seg_obs2'] = np.zeros((batch_size, mem_len, self.obs_dim))
            batch['mem_seg_act'] = np.zeros((batch_size, mem_len, self.act_dim))
            for i, id in enumerate(idxs):
                start_id = id - mem_len + 1
                if start_id < 0:
                    start_id = 0
                # If exist done before the last experience, start from the index next to the done.
                if len(np.where(self.done_buf[start_id:id]==1)[0]) != 0:
                    start_id = start_id + (np.where(self.done_buf[start_id:id] == 1)[0][-1])+1
                seg_len = id - start_id + 1
                batch['mem_seg_len'][i] = seg_len
                batch['mem_seg_obs'][i, :seg_len, :] = self.obs_buf[start_id:id+1]
                batch['mem_seg_obs2'][i, :seg_len, :] = self.obs2_buf[start_id:id+1]
                batch['mem_seg_act'][i, :seg_len, :] = self.act_buf[start_id:id+1]
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k, v in batch.items()}


class DDPG(object):
    def __init__(self, obs_space, act_space, hidden_sizes,
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

        # Create replay buffer
        self.obs = None
        self.act = None
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
        self.ac = MLPActorCritic(self.obs_dim, self.act_dim, self.act_limit, self.hidden_sizes)
        self.ac_targ = deepcopy(self.ac)

        self.ac_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ac.to(self.ac_device)
        self.ac_targ.to(self.ac_device)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q.parameters())

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

        q = self.ac.q(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            pi_targ = self.ac_targ.pi(o2)
            a2 = pi_targ

            # Target Q-values
            q_pi_targ = self.ac_targ.q(o2, a2)
            backup = r + self.gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup) ** 2).mean()

        # Useful info for logging
        loss_info = dict(QVals=q.cpu().detach().numpy())

        return loss_q, loss_info

    def _compute_loss_pi(self, data):
        """function for computing DDPG pi loss"""
        o = data['obs']
        q_pi = self.ac.q(o, self.ac.pi(o))
        return -q_pi.mean()

    def get_train_action(self, obs):
        a = self.ac.act(torch.as_tensor(obs, dtype=torch.float32).to(self.ac_device))
        a += self.act_noise * np.random.randn(self.act_dim)
        a = np.clip(a, -self.act_limit, self.act_limit)
        return a

    def get_test_action(self, obs):
        a = self.ac.act(torch.as_tensor(obs, dtype=torch.float32).to(self.ac_device))
        return a

    def interact(self, time_step, new_obs, rew, hc_rew, done, info, terminal, logger):
        # If not the initial observation, store the latest experience (obs, act, rew, new_obs, done).
        if self.obs is not None:
            self.mem_manager.store_experience(self.obs, self.act, new_obs, rew, hc_rew, done, 'DDPG_agent',
                                              obs_time=self.obs_timestamp, act_time=info['act_datetime'], obs2_time=info['obs_datetime'])

        # If terminal, start from the new episode where no previous (obs, act) exist.
        if terminal:
            self.obs = None
            self.act = None
            self.obs_timestamp = None
        else:
            # Get action:
            #   Until start_steps have elapsed, randomly sample actions
            #   from a uniform distribution for better exploration. Afterwards,
            #   use the learned policy (with some noise, via act_noise).
            if time_step > self.start_steps:
                self.act = self.get_train_action(new_obs)
            else:
                self.act = self.act_space.sample()
            self.obs = new_obs
            self.obs_timestamp = info['obs_datetime']

        # Update
        logger = self.update(time_step, logger, done)

        return self.act, logger

    def update(self, time_step, logger, done=None):
        #
        if time_step >= self.update_after and time_step % self.update_every == 0:
            for j in range(self.update_every):
                # Sample batch from replay buffer and update agent
                batch = self.mem_manager.sample_exp_batch(self.batch_size, device=self.ac_device)

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

