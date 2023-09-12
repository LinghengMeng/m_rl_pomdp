import collections
import numpy as np
import gymnasium as gym


class DecisionProcessWrapper(gym.ObservationWrapper):
    """
    DecisionProcessWrapper takes environment name, dp_type, and dp_type related hyper-parameters to decide the type of the decision process.
    (Note: call gym.pprint_registry() to print the whole registry in gymnasium.)
    """
    def __init__(self, env_name, dp_type='MDP',
                 flicker_prob=0.2, random_noise_sigma=0.1, random_sensor_missing_prob=0.1, obs_tile_num=1, obs_tile_value=None):
        """
        :param env_name:
        :param dp_type:
            1.  MDP: original task
            2.  POMDP-RV: remove velocity related observation
            3.  POMDP-FLK: obscure the entire observation with a certain probability at each time step with the
                           probability flicker_prob.
            4.  POMDP-RN: each sensor in an observation is disturbed by a random noise Normal ~ (0, sigma).
            5.  POMDP-RSM: each sensor in an observation will miss with a relatively low probability sensor_miss_prob
            6.  POMDP-RV_and_POMDP-FLK:
            7.  POMDP-RV_and_POMDP-RN:
            8.  POMDP-RV_and_POMDP-RSM:
            9.  POMDP-FLK_and_POMDP-RN:
            10. POMDP-RN_and_POMDP-RSM:
            11. POMDP-RSM_and_POMDP-RN:

        """
        super().__init__(gym.make(env_name))
        self.dp_type = dp_type
        self.flicker_prob = flicker_prob
        self.random_noise_sigma = random_noise_sigma
        self.random_sensor_missing_prob = random_sensor_missing_prob
        self.obs_tile_num = obs_tile_num
        self.obs_tile_value = obs_tile_value
        self._max_episode_steps = self.env._max_episode_steps

        if self.dp_type == 'MDP':
            pass
        elif self.dp_type == 'POMDP-RV':
            # Remove Velocity info, comes with the change in observation space.
            self.remain_obs_idx, self.observation_space = self._remove_velocity(env_name)
        elif self.dp_type == 'POMDP-FLK':
            pass
        elif self.dp_type == 'POMDP-RN':
            pass
        elif self.dp_type == 'POMDP-RSM':
            pass
        elif self.dp_type == 'POMDP-RV_and_POMDP-FLK':
            # Remove Velocity Info, comes with the change in observation space.
            self.remain_obs_idx, self.observation_space = self._remove_velocity(env_name)
        elif self.dp_type == 'POMDP-RV_and_POMDP-RN':
            # Remove Velocity Info, comes with the change in observation space.
            self.remain_obs_idx, self.observation_space = self._remove_velocity(env_name)
        elif self.dp_type == 'POMDP-RV_and_POMDP-RSM':
            # Remove Velocity Info, comes with the change in observation space.
            self.remain_obs_idx, self.observation_space = self._remove_velocity(env_name)
        elif self.dp_type == 'POMDP-FLK_and_POMDP-RN':
            pass
        elif self.dp_type == 'POMDP-RN_and_POMDP-RSM':
            pass
        elif self.dp_type == 'POMDP-RSM_and_POMDP-RN':
            pass
        else:
            raise ValueError("dp_type={} was incorrect!".format(self.dp_type))

        # Add tile observation to test unbalanced observation space and action space
        if self.obs_tile_num > 1:
            tiled_obs_dim = self.observation_space.shape[0] * int(self.obs_tile_num)
            obs_low = np.array([-np.inf for i in range(tiled_obs_dim)], dtype="float32")
            obs_high = np.array([np.inf for i in range(tiled_obs_dim)], dtype="float32")
            self.observation_space = gym.spaces.Box(obs_low, obs_high)

    def observation(self, obs):
        # Single source of POMDP
        if self.dp_type == 'MDP':
            new_obs = obs
        elif self.dp_type == 'POMDP-RV':
            new_obs = obs.flatten()[self.remain_obs_idx]
        elif self.dp_type == 'POMDP-FLK':
            # Note: POMDP-FLK is equivalent to:
            #   POMDP-FLK_and_POMDP-RSM, POMDP-RN_and_POMDP-FLK, POMDP-RSM_and_POMDP-FLK
            if np.random.rand() <= self.flicker_prob:
                new_obs = np.zeros(obs.shape)
            else:
                new_obs = obs.flatten()
        elif self.dp_type == 'POMDP-RN':
            new_obs = (obs + np.random.normal(0, self.random_noise_sigma, obs.shape)).flatten()
        elif self.dp_type == 'POMDP-RSM':
            obs[np.random.rand(len(obs)) <= self.random_sensor_missing_prob] = 0
            new_obs = obs.flatten()
        # Multiple source of POMDP
        elif self.dp_type == 'POMDP-RV_and_POMDP-FLK':
            # Note: POMDP-RV_and_POMDP-FLK is equivalent to POMDP-FLK_and_POMDP-RV
            # Remove velocity
            new_obs = obs.flatten()[self.remain_obs_idx]
            # Flickering
            if np.random.rand() <= self.flicker_prob:
                new_obs = np.zeros(new_obs.shape)
            else:
                new_obs = new_obs
        elif self.dp_type == 'POMDP-RV_and_POMDP-RN':
            # Note: POMDP-RV_and_POMDP-RN is equivalent to POMDP-RN_and_POMDP-RV
            # Remove velocity
            new_obs = obs.flatten()[self.remain_obs_idx]
            # Add random noise
            new_obs = (new_obs + np.random.normal(0, self.random_noise_sigma, new_obs.shape)).flatten()
        elif self.dp_type == 'POMDP-RV_and_POMDP-RSM':
            # Note: POMDP-RV_and_POMDP-RSM is equivalent to POMDP-RSM_and_POMDP-RV
            # Remove velocity
            new_obs = obs.flatten()[self.remain_obs_idx]
            # Random sensor missing
            new_obs[np.random.rand(len(new_obs)) <= self.random_sensor_missing_prob] = 0
        elif self.dp_type == 'POMDP-FLK_and_POMDP-RN':
            # Flickering
            if np.random.rand() <= self.flicker_prob:
                new_obs = np.zeros(obs.shape)
            else:
                new_obs = obs
            # Add random noise
            new_obs = (new_obs + np.random.normal(0, self.random_noise_sigma, new_obs.shape)).flatten()
        elif self.dp_type == 'POMDP-RN_and_POMDP-RSM':
            # Random noise
            new_obs = (obs + np.random.normal(0, self.random_noise_sigma, obs.shape)).flatten()
            # Random sensor missing
            new_obs[np.random.rand(len(new_obs)) <= self.random_sensor_missing_prob] = 0
        elif self.dp_type == 'POMDP-RSM_and_POMDP-RN':
            # Random sensor missing
            obs[np.random.rand(len(obs)) <= self.random_sensor_missing_prob] = 0
            # Random noise
            new_obs = (obs + np.random.normal(0, self.random_noise_sigma, obs.shape)).flatten()
        else:
            raise ValueError("pomdp_type was not in ['MDP','POMDP-RV','POMDP-FLK','POMDP-RN','POMDP-RSM',"
                             "'POMDP-RV_and_POMDP-FLK','POMDP-RV_and_POMDP-RN','POMDP-RV_and_POMDP-RSM',"
                             "'POMDP-FLK_and_POMDP-RN','POMDP-RN_and_POMDP-RSM','POMDP-RSM_and_POMDP-RN']!")

        # Add tile observation to test unbalanced observation space and action space
        if self.obs_tile_num > 1:
            if self.obs_tile_value is None:
                new_obs = np.tile(new_obs, self.obs_tile_num)
            else:
                # Tile with a given value
                new_obs = np.concatenate((new_obs, self.obs_tile_value*np.ones(len(new_obs)*(self.obs_tile_num-1))))
        return new_obs

    def _remove_velocity(self, env_name):
        # OpenAIGym
        #  1. Classic Control
        if env_name == "Pendulum-v1":
            remain_obs_idx = np.arange(0, 2)          # angular velocity entry id: 2 (https://gymnasium.farama.org/environments/classic_control/pendulum/)
        elif env_name == "MountainCarContinuous-v0":
            remain_obs_idx = list([0])                # angular velocity entry id: 1 (https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/)
        #  1. MuJoCo
        elif env_name == "HalfCheetah-v4":
            remain_obs_idx = list(np.arange(0, 4)) + list(np.arange(8, 13))  # angular velocity entry id: 4,5,6,7,13,14,15,16 (https://gymnasium.farama.org/environments/mujoco/half_cheetah/)
        elif env_name == "Ant-v4":
            remain_obs_idx = list(np.arange(0, 13))                          # angular velocity entry id: 13-26 (https://gymnasium.farama.org/environments/mujoco/ant/)
        elif env_name == 'Walker2d-v4':
            remain_obs_idx = np.arange(0, 8)                                 # angular velocity entry id: 8-16 (https://gymnasium.farama.org/environments/mujoco/walker2d/)
        elif env_name == 'Hopper-v4':
            remain_obs_idx = np.arange(0, 5)                                 # angular velocity entry id: 5-10（https://gymnasium.farama.org/environments/mujoco/hopper/）
        elif env_name == "InvertedPendulum-v4":
            remain_obs_idx = np.arange(0, 2)                                 # angular velocity entry id: 2,3 (https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/)
        elif env_name == "InvertedDoublePendulum-v4":
            remain_obs_idx = list(np.arange(0, 5)) + list(np.arange(8, 11))  # angular velocity entry id: 5-7 (https://gymnasium.farama.org/environments/mujoco/inverted_double_pendulum/)
        elif env_name == "Swimmer-v4":
            remain_obs_idx = np.arange(0, 3)                                 # angular velocity entry id: 3-7 (https://gymnasium.farama.org/environments/mujoco/swimmer/)
        elif env_name == "Pusher-v4":
            remain_obs_idx = list(np.arange(0, 7)) + list(np.arange(14, 23)) # angular velocity entry id: 7-13 (https://gymnasium.farama.org/environments/mujoco/pusher/)
        elif env_name == "Reacher-v4":
            remain_obs_idx = list(np.arange(0, 6)) + list(np.arange(8, 11))  # angular velocity entry id: 6,7 (https://gymnasium.farama.org/environments/mujoco/reacher/)
        elif env_name == 'Humanoid-v4':
            remain_obs_idx = list(np.arange(0, 22)) + list(np.arange(45, 185)) + list(np.arange(269, 376)) # angular velocity entry id: 22-44, 185-268 (https://gymnasium.farama.org/environments/mujoco/humanoid/)
        elif env_name == 'HumanoidStandup-v4':
            remain_obs_idx = list(np.arange(0, 22)) + list(np.arange(45, 185)) + list(np.arange(269, 376)) # angular velocity entry id: 22-44, 185-268 (https://gymnasium.farama.org/environments/mujoco/humanoid_standup/)
        # 2. MuJoCo v2 and v3 previously developed by OpenAI Gym (verified by inspecting source code in https://github.com/openai/gym/tree/master/gym/envs/mujoco)
        elif env_name == "HalfCheetah-v3" or env_name == "HalfCheetah-v2":
            remain_obs_idx = list(np.arange(0, 8)) 
        elif env_name == "Ant-v3" or env_name == "Ant-v2":
            remain_obs_idx = list(np.arange(0, 13)) + list(np.arange(27, 111))     
        elif env_name == 'Walker2d-v3' or env_name == "Walker2d-v2":
            remain_obs_idx = np.arange(0, 8)                                        
        elif env_name == 'Hopper-v3' or env_name == "Hopper-v2":
            remain_obs_idx = np.arange(0, 5)                               
        elif env_name == "InvertedPendulum-v2":
            remain_obs_idx = np.arange(0, 2)                                
        elif env_name == "InvertedDoublePendulum-v2":
            remain_obs_idx = list(np.arange(0, 5)) + list(np.arange(8, 11))  
        elif env_name == "Swimmer-v3" or env_name == "Swimmer-v2":
            remain_obs_idx = np.arange(0, 3)                                
        elif env_name == "Thrower-v2":
            remain_obs_idx = list(np.arange(0, 7)) + list(np.arange(14, 23))
        elif env_name == "Striker-v2":
            remain_obs_idx = list(np.arange(0, 7)) + list(np.arange(14, 23))
        elif env_name == "Pusher-v2":
            remain_obs_idx = list(np.arange(0, 7)) + list(np.arange(14, 23))
        elif env_name == "Reacher-v2":
            remain_obs_idx = list(np.arange(0, 6)) + list(np.arange(8, 11))
        elif env_name == 'Humanoid-v3' or env_name == "Humanoid-v2":
            remain_obs_idx = list(np.arange(0, 22)) + list(np.arange(45, 185)) + list(np.arange(269, 376))
        elif env_name == 'HumanoidStandup-v2':
            remain_obs_idx = list(np.arange(0, 22)) + list(np.arange(45, 185)) + list(np.arange(269, 376))
        else:
            raise ValueError('POMDP for {} is not defined!'.format(env_name))

        # Redefine observation_space
        obs_low = np.array([-np.inf for i in range(len(remain_obs_idx))], dtype="float32")
        obs_high = np.array([np.inf for i in range(len(remain_obs_idx))], dtype="float32")
        observation_space = gym.spaces.Box(obs_low, obs_high)
        return remain_obs_idx, observation_space

    def reset(self, seed):
        observation, info = self.env.reset(seed=seed)
        return self.observation(observation), info


if __name__ == '__main__':

    env_name_list = ["Pendulum-v1", "MountainCarContinuous-v0",
                     "HalfCheetah-v4", "Ant-v4", 'Walker2d-v4', 'Hopper-v4',
                     "InvertedPendulum-v4", "InvertedDoublePendulum-v4",
                     "Swimmer-v4", "Pusher-v4", "Reacher-v4",
                     'Humanoid-v4', 'HumanoidStandup-v4']
    dp_type_list = ['MDP','POMDP-RV','POMDP-FLK','POMDP-RN','POMDP-RSM',
                    'POMDP-RV_and_POMDP-FLK','POMDP-RV_and_POMDP-RN','POMDP-RV_and_POMDP-RSM',
                    'POMDP-FLK_and_POMDP-RN','POMDP-RN_and_POMDP-RSM','POMDP-RSM_and_POMDP-RN']

    for env_name in env_name_list:
        print('Env name: {}'.format(env_name))
        for dp_type in dp_type_list:
            print('\tDecision process type: {}'.format(dp_type))
            env = DecisionProcessWrapper(env_name, dp_type)
            obs, info = env.reset()
            env.step(env.action_space.sample())
            print('\t\taction_space dim: {}'.format(env.action_space.shape[0]))
            print('\t\tobservation_space dim: {}'.format(env.observation_space.shape[0]))
            print('\t\taction_space: {}'.format(env.action_space))
            print('\t\tobservation_space: {}'.format(env.observation_space))
            print('\t\tobs_length={}'.format(len(obs)))


