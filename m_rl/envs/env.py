from m_rl.envs.dp_wrapper import DecisionProcessWrapper

def make_gym_task(env_id, dp_type='MDP',
                  flicker_prob=0.2,
                  random_noise_sigma=0.1,
                  random_sensor_missing_prob=0.1,
                  fps=40, render_width=640, render_height=480, obs_tile_num=1, obs_tile_value=None):
    """

    :param env_id:
    :param dp_type: ['MDP', 'POMDP-RV', 'POMDP-FLK', 'POMDP-RN', 'POMDP-RSM']
    :param flicker_prob:
    :param random_noise_sigma:
    :param random_sensor_missing_prob:
    :param fps:
    :param render_width:
    :param render_height:
    :return:
    """
    # DecisionProcessWrapper is a warpper to implement POMDP tasks
    # dp_type is in {"MDP", 'POMDP-RV', 'POMDP-FLK', 'POMDP-RN', 'POMDP-RSM'}
    env = DecisionProcessWrapper(env_id, dp_type=dp_type,
                                 flicker_prob=flicker_prob, random_noise_sigma=random_noise_sigma,
                                 random_sensor_missing_prob=random_sensor_missing_prob, obs_tile_num=obs_tile_num, obs_tile_value=obs_tile_value)
    return env


if __name__ == '__main__':
    env = make_gym_task("Ant-v4")
    observation, info = env.reset()
    act = env.action_space.sample()
    print('act={}'.format(act))
    observation, reward, terminated, truncated, info = env.step(act)
    print('observation={}'.format(observation))
