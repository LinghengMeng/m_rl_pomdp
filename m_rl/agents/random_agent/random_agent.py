import os


class RandomAgent(object):
    def __init__(self, obs_space, act_space,
                 mem_manager=None, checkpoint_dir=None):
        #
        self.obs_space = obs_space
        self.act_space = act_space
        self.obs_dim = obs_space.shape[0]
        self.act_dim = act_space.shape[0]
        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_limit = act_space.high[0]

        # Create replay buffer
        self.obs = None
        self.act = None
        self.mem_manager = mem_manager
        assert checkpoint_dir is not None, "Checkpoint_dir is None!"
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.cp_dir = checkpoint_dir

    def save_checkpoint(self):
        """Save learned reward network to disk."""
        save_elements = {}
        return save_elements

    def restore_checkpoint(self, restore_elements, mem_manager):
        print('Successfully restored Agent!')
        self.mem_manager = mem_manager

    def get_train_action(self, obs):
        a = self.act_space.sample()
        return a

    def get_test_action(self, obs):
        a = self.act_space.sample()
        return a

    def interact(self, time_step, new_obs, rew, hc_rew, done, info, terminal, logger):
        # If not the initial observation, store the latest experience (obs, act, rew, new_obs, done).
        if self.obs is not None:
            self.mem_manager.store_experience(self.obs, self.act, new_obs, rew, hc_rew, done, 'random_agent',
                                              obs_time=self.obs_timestamp, act_time=info['act_datetime'], obs2_time=info['obs_datetime'])

        # If terminal, start from the new episode where no previous (obs, act) exist.
        if terminal:
            self.obs = None
            self.act = None
            self.obs_timestamp = None
        else:
            # Get action:
            self.act = self.get_train_action(new_obs)
            self.obs = new_obs
            self.obs_timestamp = info['obs_datetime']

        return self.act, logger