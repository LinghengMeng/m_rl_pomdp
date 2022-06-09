import numpy as np
import torch


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class RamReplayBuffer:
    """
    A simple FIFO experience replay buffer for TD3 agents.
    """

    def __init__(self, obs_dim, act_dim, max_replay_size=1e6):
        # Experience replay buffer with longer history
        obs_dim = int(obs_dim)
        act_dim = int(act_dim)
        max_replay_size = int(max_replay_size)
        self.obs_buf = np.zeros(combined_shape(max_replay_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(max_replay_size, act_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(max_replay_size, obs_dim), dtype=np.float32)
        self.pb_rew_buf = np.zeros(max_replay_size, dtype=np.float32)
        self.hc_rew_buf = np.zeros(max_replay_size, dtype=np.float32)
        self.done_buf = np.zeros(max_replay_size, dtype=np.float32)
        self.ptr, self.size, self.replay_size = 0, 0, max_replay_size

    def store(self, obs, act, next_obs, pb_rew, hc_rew, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.pb_rew_buf[self.ptr] = pb_rew
        self.hc_rew_buf[self.ptr] = hc_rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.replay_size
        self.size = min(self.size+1, self.replay_size)

    def sample_batch(self, batch_size=32, device=None):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.pb_rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k, v in batch.items()}


class RamSegmentBuffer:
    """

    """
    def __init__(self, max_buffer_size=200):
        self.obs_traj = []
        self.act_traj = []
        self.obs2_traj = []
        self.orig_rew_traj = []
        self.done_traj = []
        self.seg_length = []
        self.sampled_num = []
        self.buffer_size = 0
        self.ptr = 0
        self.max_buffer_size=max_buffer_size

    @property
    def collected_seg_num(self):
        return self.buffer_size

    def store(self, seg):
        if self.buffer_size==self.max_buffer_size:
            self.obs_traj.pop(0)
            self.act_traj.pop(0)
            self.obs2_traj.pop(0)
            self.orig_rew_traj.pop(0)
            self.done_traj.pop(0)
            self.seg_length.pop(0)
            self.sampled_num.pop(0)
        else:
            self.buffer_size += 1

        self.obs_traj.append(seg['obs_traj'])
        self.act_traj.append(seg['act_traj'])
        self.obs2_traj.append(seg['obs2_traj'])
        self.orig_rew_traj.append(seg['orig_rew_traj'])
        self.done_traj.append(seg['done_traj'])
        self.seg_length.append(seg['seg_len'])
        self.sampled_num.append(0)


    def sample_segment(self, seg_id):
        seg_id -= 1  # Different from database, here id starts from 0.
        seg = {}
        seg['id'] = seg_id
        seg['obs_traj'] = self.obs_traj[seg_id]
        seg['act_traj'] = self.act_traj[seg_id]
        seg['obs2_traj'] = self.obs2_traj[seg_id]
        seg['orig_rew_traj'] = self.orig_rew_traj[seg_id]
        seg['done_traj'] = self.done_traj[seg_id]
        seg['seg_length'] = self.seg_length[seg_id]
        seg['sampled_num'] = self.sampled_num[seg_id]
        self.sampled_num[seg_id] += 1
        return seg

class RamPreferenceBuffer:
    """

    """
    def __init__(self):
        self.train_set_left_seg_id = []
        self.train_set_left_seg_obs_traj = []
        self.train_set_left_seg_act_traj = []
        self.train_set_left_seg_obs2_traj = []
        self.train_set_left_seg_orig_rew_traj = []
        self.train_set_left_seg_done_traj = []
        self.train_set_left_seg_length = []
        self.train_set_left_seg_sampled_num = []
        self.train_set_right_seg_id = []
        self.train_set_right_seg_obs_traj = []
        self.train_set_right_seg_act_traj = []
        self.train_set_right_seg_obs2_traj = []
        self.train_set_right_seg_orig_rew_traj = []
        self.train_set_right_seg_done_traj = []
        self.train_set_right_seg_length = []
        self.train_set_pref_label = []
        self.train_set_sampled_num = []
        self.train_set_size = 0

        self.test_set_left_seg_id = []
        self.test_set_left_seg_obs_traj = []
        self.test_set_left_seg_act_traj = []
        self.test_set_left_seg_obs2_traj = []
        self.test_set_left_seg_orig_rew_traj = []
        self.test_set_left_seg_done_traj = []
        self.test_set_left_seg_length = []
        self.test_set_left_seg_sampled_num = []
        self.test_set_right_seg_id = []
        self.test_set_right_seg_obs_traj = []
        self.test_set_right_seg_act_traj = []
        self.test_set_right_seg_obs2_traj = []
        self.test_set_right_seg_orig_rew_traj = []
        self.test_set_right_seg_done_traj = []
        self.test_set_right_seg_length = []
        self.test_set_pref_label = []
        self.test_set_sampled_num = []
        self.test_set_size = 0

    @property
    def collected_pref_num(self):
        return self.train_set_size + self.test_set_size

    @property
    def training_dataset(self):
        train_dataset = []
        for i in range(self.train_set_size):
            train_dataset.append({'left_seg_obs_traj': self.train_set_left_seg_obs_traj[i],
                                  'left_seg_act_traj': self.train_set_left_seg_act_traj[i],
                                  'left_seg_obs2_traj': self.train_set_left_seg_obs2_traj[i],
                                  'left_seg_length': self.train_set_left_seg_length[i],
                                  'right_seg_obs_traj': self.train_set_right_seg_obs_traj[i],
                                  'right_seg_act_traj': self.train_set_right_seg_act_traj[i],
                                  'right_seg_obs2_traj': self.train_set_right_seg_obs2_traj[i],
                                  'right_seg_length': self.train_set_right_seg_length[i],
                                  'pref_label': self.train_set_pref_label[i]})
        return train_dataset

    @property
    def test_dataset(self):
        test_dataset = []
        for i in range(self.test_set_size):
            test_dataset.append({'left_seg_obs_traj': self.test_set_left_seg_obs_traj[i],
                                 'left_seg_act_traj': self.test_set_left_seg_act_traj[i],
                                 'left_seg_obs2_traj': self.test_set_left_seg_obs2_traj[i],
                                 'left_seg_length': self.test_set_left_seg_length[i],
                                 'right_seg_obs_traj': self.test_set_right_seg_obs_traj[i],
                                 'right_seg_act_traj': self.test_set_right_seg_act_traj[i],
                                 'right_seg_obs2_traj': self.test_set_right_seg_obs2_traj[i],
                                 'right_seg_length': self.test_set_right_seg_length[i],
                                 'pref_label': self.test_set_pref_label[i]})
        return test_dataset

    def store(self, left_seg, right_seg, pref_label, train_set=True):
        if train_set:
            self.train_set_left_seg_id.append(left_seg['id'])
            self.train_set_left_seg_obs_traj.append(left_seg['obs_traj'])
            self.train_set_left_seg_act_traj.append(left_seg['act_traj'])
            self.train_set_left_seg_obs2_traj.append(left_seg['obs2_traj'])
            self.train_set_left_seg_orig_rew_traj.append(left_seg['orig_rew_traj'])
            self.train_set_left_seg_done_traj.append(left_seg['done_traj'])
            self.train_set_left_seg_length.append(left_seg['seg_length'])
            self.train_set_left_seg_sampled_num.append(left_seg['sampled_num'])
            self.train_set_right_seg_id.append(right_seg['id'])
            self.train_set_right_seg_obs_traj.append(right_seg['obs_traj'])
            self.train_set_right_seg_act_traj.append(right_seg['act_traj'])
            self.train_set_right_seg_obs2_traj.append(right_seg['obs2_traj'])
            self.train_set_right_seg_orig_rew_traj.append(right_seg['orig_rew_traj'])
            self.train_set_right_seg_done_traj.append(right_seg['done_traj'])
            self.train_set_right_seg_length.append(right_seg['seg_length'])
            self.train_set_left_seg_sampled_num.append(right_seg['sampled_num'])
            self.train_set_pref_label.append(pref_label)
            self.train_set_sampled_num.append(0)
            self.train_set_size += 1
        else:
            self.test_set_left_seg_id.append(left_seg['id'])
            self.test_set_left_seg_obs_traj.append(left_seg['obs_traj'])
            self.test_set_left_seg_act_traj.append(left_seg['act_traj'])
            self.test_set_left_seg_obs2_traj.append(left_seg['obs2_traj'])
            self.test_set_left_seg_orig_rew_traj.append(left_seg['orig_rew_traj'])
            self.test_set_left_seg_done_traj.append(left_seg['done_traj'])
            self.test_set_left_seg_length.append(left_seg['seg_length'])
            self.test_set_left_seg_sampled_num.append(left_seg['sampled_num'])
            self.test_set_right_seg_id.append(right_seg['id'])
            self.test_set_right_seg_obs_traj.append(right_seg['obs_traj'])
            self.test_set_right_seg_act_traj.append(right_seg['act_traj'])
            self.test_set_right_seg_obs2_traj.append(right_seg['obs2_traj'])
            self.test_set_right_seg_orig_rew_traj.append(right_seg['orig_rew_traj'])
            self.test_set_right_seg_done_traj.append(right_seg['done_traj'])
            self.test_set_right_seg_length.append(right_seg['seg_length'])
            self.test_set_left_seg_sampled_num.append(right_seg['sampled_num'])
            self.test_set_pref_label.append(pref_label)
            self.test_set_sampled_num.append(0)
            self.test_set_size += 1


if __name__ == '__main__':
    import os
    obs_dim = 852
    act_dim = 6
    max_replay_size = 1e6
    ram_replay_buff = RamReplayBuffer(obs_dim, act_dim, max_replay_size)

    # Simulate experiences
    experience_num = 1000000

    obs = np.random.rand(obs_dim)
    act = np.random.rand(act_dim)
    new_obs = np.random.rand(obs_dim)
    rew = 0
    hc_rew = 0
    done = False
    for i in range(experience_num):
        if i % 10000 == 0:
            print(i)
        ram_replay_buff.store(obs, act, new_obs, rew, hc_rew, done)

    data_dir = os.path.join(
        os.path.dirname('F:/scratch/lingheng/'), 'test_db_size')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    cp_file = os.path.join(data_dir, 'test_ram_buffer_size.pt')
    save_elements = {'ram_replay_buff': ram_replay_buff}
    torch.save(save_elements, cp_file)
    import pdb;
    pdb.set_trace()