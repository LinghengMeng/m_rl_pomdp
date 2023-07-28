"""
Internal Environment

This module provides facilities for defining internal environment which is a simulation of
the internal world of an agent. To differentiate it from "External Environment", Internal
Environment is the place where intrinsic and extrinsic motivations can be defined. This is
the interface where an agent can interact with the external environment.

"""
import gymnasium as gym
import numpy as np
from datetime import datetime
import collections
from m_rl.envs.env import make_gym_task


class IntlEnv(gym.Wrapper):
    """Internal Environment is the place to define reward signal when it is provided by external environment."""
    def __init__(self, env_id, seed,
                 env_dp_type='MDP', act_transform_type=None, obs_delay_step=0,
                 render_width=640, render_height=480, obs_tile_num=1, obs_tile_value=None):
        super(IntlEnv, self).__init__(make_gym_task(env_id, dp_type=env_dp_type,
                                                    render_width=render_width, render_height=render_height,
                                                    obs_tile_num=obs_tile_num, obs_tile_value=obs_tile_value))
        self.seed = seed
        self._max_episode_steps = self.env._max_episode_steps
        # Action transformation
        self.act_transform_type = act_transform_type

        # Observation Delay
        self.obs_delay_step = obs_delay_step
        if self.obs_delay_step is None or self.obs_delay_step == 0:
            pass
        else:
            self.obs_delay_queue = collections.deque(
                maxlen=self.obs_delay_step + 1)  # The deque saves previous delay_step observations and the current observation
            self.rew_delay_queue = collections.deque(
                maxlen=self.obs_delay_step + 1)  # The deque saves previous delay_step observations and the current observation

        #
        self.obs = None           # current observation
        #
        self.obs_dim = self.observation_space.shape[0]
        self.act_dim = self.action_space.shape[0]
        self.obs_traj, self.act_traj, self.obs2_traj = [], [], []

    def render_full_obs(self, full_obs):
        return self.env.render_full_obs(full_obs)

    def step(self, act):
        # Action transformation
        if self.act_transform_type is None:
            pass
        elif self.act_transform_type == 'neg':
            act = -act
        elif self.act_transform_type == 'tanh':
            act = np.tanh(act)
        elif self.act_transform_type == 'sign_square':
            act_sign = np.ones(len(act))
            act_sign[act < 0] = -1
            act = act_sign*(act**2)
        elif self.act_transform_type == 'abs_times_2_minus_1':
            act = np.abs(act)*2-1
        elif self.act_transform_type == 'neg_abs_times_2_plus_1':
            act = (-np.abs(act))*2+1
        elif self.act_transform_type == 'random':
            act = self.action_space.sample()
        else:
            raise ValueError('act_transform_type={} is not defined!'.format(self.act_transform_type))

        # Interact with environment
        act_ts = datetime.now()  # Action execution timestamp
        obs2, orig_rew, terminated, truncated,  info = self.env.step(act)
        new_obs_ts = datetime.now()

        # Observation delay
        if self.obs_delay_step is None or self.obs_delay_step == 0:
            extl_rew = orig_rew
        else:
            self.obs_delay_queue.append(obs2)
            self.rew_delay_queue.append(orig_rew)
            if len(self.obs_delay_queue) < (self.obs_delay_step+1):
                # TODO: return random rather than zero for the first few observations
                obs2 = np.zeros(self.observation_space.shape[0])
                extl_rew = 0
            else:
                obs2 = self.obs_delay_queue.pop()
                extl_rew = self.rew_delay_queue.pop()

        # Crucial note: Update current observation after reward computation.
        self.obs = obs2
        # Store extl_env to info for diagnostic purpose
        info['extl_rew'] = extl_rew
        info['orig_rew'] = orig_rew
        info['act_datetime'] = act_ts
        info['obs_datetime'] = new_obs_ts
        info['orig_rew'] = orig_rew
        return obs2, extl_rew, terminated, truncated, info

    def reset(self):
        # Empty episode memory
        self.obs_traj, self.act_traj, self.obs2_traj = [], [], []

        # Empty obs_delay_queue and rew_delay_queue
        if self.obs_delay_step is None or self.obs_delay_step == 0:
            pass
        else:
            self.obs_delay_queue.clear()
            self.rew_delay_queue.clear()

        #
        self.obs, info = self.env.reset(self.seed)
        obs_ts = datetime.now()

        # No need to delay the first observation, because it's not a consequence of an action given by the agent

        info['obs_datetime'] = obs_ts
        return self.obs, info


if __name__ == '__main__':
    from pl.envs.env import make_gym_task
    extl_env = make_gym_task('Ant-v2')
    intl_env = IntlEnv('Ant-v2', act_transform_type='tanh')
    intl_env.reset()
    obs2, rew, done, info = intl_env.step(intl_env.action_space.sample())
