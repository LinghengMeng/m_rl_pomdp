"""
db_manager.py declares all necessary api for manipulate database tables.

sqlalchemy is used in order to make the implemented functions be database transparent.
"""
import os
from m_rl.mems.db_config import db_table_config
import numpy as np
import pandas as pd
import random
import torch
import datetime
import sqlite3
import sqlalchemy as sqla
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.types import TypeDecorator, TEXT
from sqlalchemy.orm import sessionmaker
import json

Base = declarative_base()


class JSONEncodedArray(TypeDecorator):
    """
    Represents an numpy.array as a json-encoded text.
    Reference: https://docs.sqlalchemy.org/en/14/core/custom_types.html#sqlalchemy.types.TypeDecorator
    """
    impl = TEXT
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            value = json.dumps(value.tolist())
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            value = np.asarray(json.loads(value))
        return value


def table_column_declare(table_config):
    table_columns = {}
    for column_name in table_config:
        col_data_type = table_config[column_name]["data_type"]
        col_default = table_config[column_name]["default"]
        col_primary_key = table_config[column_name]["primary_key"]
        col_foreign_key = table_config[column_name]["foreign_key"]
        # Define data_type
        if col_data_type == "int":
            data_type = sqla.Integer
        elif col_data_type == "float":
            data_type = sqla.Float
        elif col_data_type == "boolean":
            data_type == sqla.Boolean
        elif col_data_type == "text":
            data_type = sqla.Text
        elif col_data_type == "array":
            # data_type = sqla.ARRAY(sqla.Float)
            data_type = JSONEncodedArray
        elif col_data_type == "time":
            data_type = sqla.DateTime
        else:
            raise ValueError("col_data_type: {} not defined!".format())

        if column_name == "create_time":
            col_default = sqla.func.now()

        primary_key = True if col_primary_key is not None else False
        if col_foreign_key is not None:
            column = sqla.Column(data_type, sqla.ForeignKey('{}.{}'.format(col_foreign_key[0], col_foreign_key[1])),
                                 primary_key=primary_key, default=col_default)
        else:
            column = sqla.Column(data_type, primary_key=primary_key, default=col_default)
        table_columns[column_name] = column
    return table_columns


###########################################################################################
#                               Declare Database Tables                                   #
###########################################################################################
class ExperienceTable(Base):
    __tablename__ = "experience_table"
    table_columns = table_column_declare(db_table_config[__tablename__])
    # Convert to local variables
    for column_name in table_columns:
        vars()[column_name] = table_columns[column_name]


###########################################################################################
#                         Declare Database Table Operators                                #
###########################################################################################
class ExperienceTableOperator:
    """A replay buffer based on database."""

    def __init__(self, db_session, max_replay_size=1e6, table_name="experience"):
        """Initialize replay buffer"""

        self.max_replay_size = max_replay_size
        self.db_not_committed = False           # Indicating if commit() needs to be called.

        # Connect to database
        self.db_session = db_session

        # Initialize current replay buffer size if the number of previous experiences in database is larger than the
        # maximum of the reply buffer size set the current replay buffer to max_replay_size.
        self.replay_buffer_size = min(self.max_replay_size, self.db_session.query(ExperienceTable).count())
        self.start_id = 0
        max_id = self.db_session.query(sqla.func.max(ExperienceTable.id)).scalar()
        if max_id is None:
            self.end_id = 0
            self.replay_buffer_size = 0
            self.start_id = 1  # start_id always start from 1 in database table
        else:
            self.end_id = max_id
            self.replay_buffer_size = min(self.max_replay_size, max_id)
            self.start_id = self.end_id - self.replay_buffer_size + 1

    def store(self, obs, act, obs2, pb_rew, hc_rew, done, behavior_mode=None, obs_time=None, act_time=None, obs2_time=None):
        """Store experience into database"""
        experience = ExperienceTable(obs=obs, act=act, obs2=obs2, pb_rew=pb_rew, hc_rew=hc_rew, done=done, behavior_mode=behavior_mode,
                                     obs_time=obs_time, act_time=act_time, obs2_time=obs2_time)
        self.db_session.add(experience)

        # Commit is time-consuming, so only commit() when done or when updating the RL-agent in sample_batch.
        if done:
            self.db_session.commit()
            self.db_not_committed = False
        else:
            self.db_not_committed = True

        if self.replay_buffer_size < self.max_replay_size:
            self.replay_buffer_size += 1
            self.end_id += 1
        else:
            self.end_id += 1
            self.start_id += 1

    def sample_batch(self, batch_size=64, device=None, reward_mem_len=None, agent_mem_len=None, multistep_size=None):
        """
        Sample a mini-batch of experiences, where:
            reward_mem_len: the length of memory for reward component
            agent_mem_len: the length of memory for agent.
            These two memory lengths are not necessarily to be the same.
        """
        # Commit if not.
        if self.db_not_committed:
            self.db_session.commit()
            self.db_not_committed = False

        # It's faster to use randint to generate random sample indices rather than fetching ids from the database.
        # np.arange(start_id, end_id) generate integers from start_id to (end_id-1), so we used (self.end_id+1).
        batch_idxs = [int(id_) for id_ in random.sample(list(np.arange(self.start_id, self.end_id + 1, dtype=int)), min(batch_size, int(self.replay_buffer_size)))]  # Random sample without replacement

        # Fetch sampled experiences from database
        batch_df = pd.read_sql(self.db_session.query(ExperienceTable).filter(ExperienceTable.id.in_(batch_idxs)).statement, self.db_session.bind)

        # To speed up, do not update sampled number
        # # Increase the sampled_num of the sampled experiences
        # self.db_session.query(ExperienceTable).filter(ExperienceTable.id.in_(batch_idxs)).update({"sampled_num": ExperienceTable.sampled_num + 1})
        # self.db_session.commit()

        # Form batch tensor: used for non-memory learning
        batch = dict(obs=np.stack(batch_df['obs']),
                     act=np.stack(batch_df['act']),
                     obs2=np.stack(batch_df['obs2']),
                     rew=np.stack(batch_df['pb_rew']),
                     hc_rew=np.stack(batch_df['hc_rew']),
                     done=np.stack(batch_df['done']))
        ####################################################
        #        Retrieve data for multistep method        #
        ####################################################
        if multistep_size is not None:
            # When multistep_size=1, this is 1-step method which just uses one step experience.
            if multistep_size < 1:
                raise ValueError('multistep_size={}, which should be always >=1.'.format(multistep_size))
            # Retrieve each step into a dataframe, rather than retrieve for each sample and cause larger for loop.
            # np.minimum(np.asarray(batch_idxs) + step_i, self.end_id): to make sure no id beyond self.end_id
            # Note: This is a possibility that multiple sample will have the sample id=self.end_id.
            multistep_df_list = []
            for step_i in range(multistep_size):
                # If batch_id+step_id > self.end_id, no data will return from database for that index.
                tmp_multistep_df = pd.read_sql(
                    self.db_session.query(ExperienceTable).filter(
                        ExperienceTable.id.in_(np.minimum((np.asarray(batch_idxs) + step_i), self.end_id).tolist())).statement,
                    self.db_session.bind)
                # The only case that no data is retrieved for an index is the index is beyond the self.end_id. Because the db return is ranked by id,
                # the missed data can be easily appended at the end using the data corresponds to self.end_id. And if we set multistep_size correctly
                # the appended experiences will not be used at all, but is just for match the data size for concatenation.
                if len(tmp_multistep_df) != len(batch_idxs):
                    for _ in range(len(batch_idxs)-len(tmp_multistep_df)):
                        tmp_multistep_df = tmp_multistep_df.append(multistep_df_list[step_i-1].iloc[-1])
                    tmp_multistep_df.reset_index(inplace=True, drop=True)    # Drop the pd index
                multistep_df_list.append(tmp_multistep_df)
            # Concatenate multisteps
            multistep_df = pd.concat(multistep_df_list, axis=1)

            # Retrieve rewards within the multistep window
            batch['multistep_reward_seg'] = multistep_df['pb_rew'].to_numpy()
            # Take minimum between multistep_size and (self.end_id - np.asarray(batch_idxs) + 1) to take into account of cases where
            # batch_idxs is less than multistep_size towards self.end_id.
            batch['multistep_size'] = np.ones(len(batch_idxs)) * np.minimum(multistep_size, self.end_id - np.asarray(batch_idxs) + 1)
            find_done_x, find_done_y = np.where(multistep_df['done'].to_numpy() == 1)  # Find done within the multistep window
            if len(find_done_x) != 0:
                # Index in done_y start from 0, so +1 to make the multistep_size start from 1.
                # Take the minimum is to consider cases where multiple done within one multistep window and we take the first one, i.e. the minimum one.
                batch['multistep_size'][find_done_x] = np.minimum(batch['multistep_size'][find_done_x], find_done_y + 1)

                # Set reward after the 1st done within a multistep window to 0.
                for x in find_done_x:
                    batch['multistep_reward_seg'][x, (batch['multistep_size'][x].astype(int) - 1):] = 0

            # Concatenate obs2 within multistep window and retrieve the last one within multistep window.
            multistep_obs2 = np.stack([np.stack(multistep_df['obs2'].values[:, m]) for m in range(multistep_size)], axis=2)
            batch['multistep_obs2'] = multistep_obs2[np.arange(len(batch['multistep_size'])), :, list(batch['multistep_size'].astype(int) - 1)]
            # Retrieve the done for the last experience within the multistep window
            batch['multistep_done'] = multistep_df['done'].to_numpy()[
                np.arange(len(batch['multistep_size'])), list(batch['multistep_size'].astype(int) - 1)]

        # Extract memory if mem_len is not None
        if reward_mem_len is not None or agent_mem_len is not None:
            ##############################################################
            #       Retrieve data for Reward and Agent with Memory       #
            ##############################################################
            # Note: for agent memory, memory length indicates the number of experiences before the current experiences, so we need to sample
            #   agent_mem_len + 1.
            if reward_mem_len is not None and agent_mem_len is not None:
                mem_len = max(reward_mem_len, agent_mem_len + 1)    # If both need memory, take the larger one, so only need to do DB request once.
            else:
                if reward_mem_len is not None:
                    mem_len = reward_mem_len
                else:
                    mem_len = agent_mem_len + 1
            # Retrieve experience step into a dataframe, rather than retrieve for each sample and cause larger for loop.
            # If id is less than self.start_id, np.minimum(np.asarray(batch_idxs) + step_i, self.end_id): to make sure no id beyond self.end_id
            # Note: This is a possibility that multiple sample will have the sample id=self.end_id.
            mem_df_list = []
            for m_i in range(mem_len):
                tmp_mem_df = pd.read_sql(self.db_session.query(ExperienceTable).filter(ExperienceTable.id.in_(np.maximum((np.asarray(batch_idxs) - m_i), self.start_id).tolist())).statement, self.db_session.bind)
                # The once case where  len(tmp_multistep_df) != len(batch_idxs) is index before self.start_id is requested. We only need to append
                # experience corresponds to self.start_id at the beginning of the dataframe.
                if len(tmp_mem_df) != len(batch_idxs):
                    for _ in range(len(batch_idxs)-len(tmp_mem_df)):
                        tmp_mem_df = pd.concat([pd.DataFrame(mem_df_list[m_i-1].iloc[0]).transpose(), tmp_mem_df])
                    tmp_mem_df.reset_index(inplace=True, drop=True)    # Drop the pd index
                mem_df_list.append(tmp_mem_df)

            # Concatenate memory
            mem_df_list.reverse()    # Note: reverse order to keep the experience order that older experience in front
            mem_df = pd.concat(mem_df_list, axis=1)

            ################################################
            #                 Reward Memory                #
            ################################################
            if reward_mem_len is not None:
                # Take minimum between reward_mem_len and (np.asarray(batch_idxs) - self.start_id) as initial value to take into account of
                # cases where batch_idxs is less than reward_mem_len away from self.start_id.
                batch['reward_mem_seg_len'] = np.ones(len(batch_idxs)) * np.minimum(reward_mem_len, np.asarray(batch_idxs) - self.start_id)
                # Find done within the memory window excluding the current experience
                find_done_x, find_done_y = np.where(mem_df['done'].to_numpy()[:, :-1] == 1)

                if len(find_done_x) != 0:
                    # Important: If there is done, except the current experience within a fixed memory window, start only after the latest done,
                    # so there is a '+1'. Take the minimum with np.minimum() is to consider cases where multiple done within one memory window
                    # and we take the last one, i.e. the one will cause minimum length.
                    batch['reward_mem_seg_len'][find_done_x] = np.minimum(batch['reward_mem_seg_len'][find_done_x], reward_mem_len - (find_done_y + 1))

                if reward_mem_len == mem_len:
                    stack_start_idx = 0
                    stack_end_idx = reward_mem_len
                else:
                    # This means reward_mem_len < mem_len
                    stack_start_idx = mem_len - 1 - (reward_mem_len - 1)    # For Range(start, stop), the stop will not be included.
                    stack_end_idx = mem_len                                 # Include the current experience
                batch['reward_mem_seg_obs'] = np.stack([np.stack(mem_df['obs'].values[:, m]) for m in range(stack_start_idx, stack_end_idx)], axis=1)
                batch['reward_mem_seg_act'] = np.stack([np.stack(mem_df['act'].values[:, m]) for m in range(stack_start_idx, stack_end_idx)], axis=1)
                batch['reward_mem_seg_obs2'] = np.stack([np.stack(mem_df['obs2'].values[:, m]) for m in range(stack_start_idx, stack_end_idx)], axis=1)
            ################################################
            #                Agent Memory                  #
            ################################################
            # Note: for agent memory, the memory corresponds to the agent_mem_len experiences before the current experience, which is different
            #   from that for reward memory which also includes the current experience.
            #   For example,
            if agent_mem_len is not None and agent_mem_len > 0:
                # Take minimum between agent_mem_len and (np.asarray(batch_idxs) - self.start_id) as initial value to take into account of
                # cases where batch_idxs is less than agent_mem_len away from self.start_id.
                # Note: If batch_id == self.start_id, memory length = 0. This is because we do not consider the current experience as memory.
                batch['agent_mem_seg_len'] = np.ones(len(batch_idxs)) * np.minimum(agent_mem_len, np.asarray(batch_idxs) - self.start_id)
                find_done_x, find_done_y = np.where(mem_df['done'].to_numpy()[:, :-1] == 1)  # Find done within the memory window, i.e. exclude the current experience
                if len(find_done_x) != 0:
                    # Important: If there is done, except the current experience within a fixed memory window, start only after the latest done,
                    # so there is a '+1' and np.minimum() if there are multiple done within the memory window.
                    # Take the minimum is to consider cases where multiple done within one memory window and we take the last one, i.e. the one
                    # will cause minimum length.
                    batch['agent_mem_seg_len'][find_done_x] = np.minimum(batch['agent_mem_seg_len'][find_done_x], agent_mem_len - (find_done_y + 1))

                # Concatenate experiences before the current experience but still within memory window.
                # agent_mem_stack_id = np.stack([np.stack(mem_df['id'].values[:, m]) for m in range(agent_mem_len)], axis=1)
                if (agent_mem_len + 1) == mem_len:
                    stack_start_idx = 0
                    stack_end_idx = agent_mem_len
                else:
                    # This means (agent_mem_len + 1) < mem_len
                    stack_start_idx = mem_len - 1 -1 - (agent_mem_len - 1)    # For Range(start, stop), the stop will not be included.
                    stack_end_idx = mem_len - 1                                # Exclude the current experience
                batch['agent_mem_seg_obs'] = np.stack([np.stack(mem_df['obs'].values[:, m]) for m in range(stack_start_idx, stack_end_idx)], axis=1)
                batch['agent_mem_seg_act'] = np.stack([np.stack(mem_df['act'].values[:, m]) for m in range(stack_start_idx, stack_end_idx)], axis=1)
                batch['agent_mem_seg_obs2'] = np.stack([np.stack(mem_df['obs2'].values[:, m]) for m in range(stack_start_idx, stack_end_idx)], axis=1)
                batch['agent_mem_seg_act2'] = np.stack([np.stack(mem_df['act'].values[:, m]) for m in range(stack_start_idx+1, stack_end_idx+1)], axis=1)

        batch_tensor = {}
        for k, v in batch.items():
            if v is None:
                batch_tensor[k] = None
            else:
                # seg_len is used in pack_padded_sequence and should be on CPU, while others can be on GPU
                if k == 'reward_mem_seg_len':
                    batch_tensor[k] = torch.as_tensor(v, dtype=torch.float32)
                else:
                    batch_tensor[k] = torch.as_tensor(v, dtype=torch.float32).to(device)
        return batch_tensor

    @property
    def last_experience_id(self):
        max_id = self.db_session.query(sqla.func.max(ExperienceTable.id)).scalar()
        return max_id

    def retrieve_last_experience_episode(self, episode_exp_start_id):
        # Commit if not.
        if self.db_not_committed:
            self.db_session.commit()
            self.db_not_committed = False

        last_experience_id = self.last_experience_id
        if episode_exp_start_id is None:
            new_episode_exp_start_id = 1
        else:
            new_episode_exp_start_id = last_experience_id + 1

        if episode_exp_start_id is None:
            last_path = None
        else:
            if episode_exp_start_id == last_experience_id or episode_exp_start_id > last_experience_id:
                import pdb; pdb.set_trace()
            last_episode = pd.read_sql(
                self.db_session.query(ExperienceTable).filter(ExperienceTable.id.between(episode_exp_start_id, last_experience_id)).statement,
                self.db_session.bind)
            last_path = {"exp_id_traj":  last_episode['id'].values,
                         "human_obs_traj": np.zeros(len(last_episode['id'].values)), "behavior_mode": last_episode['behavior_mode'].values[0]}

        return last_path, new_episode_exp_start_id

    def extract_id_corresponding_time_interval(self, start_datetime, end_datetime):
        exp_df = pd.read_sql(self.db_session.query(ExperienceTable).filter(sqla.and_(start_datetime <= ExperienceTable.obs_time, ExperienceTable.obs2_time <= end_datetime)).statement, self.db_session.bind)
        if len(exp_df) == 0:
            return None, None
        else:
            return exp_df['id'].values[0], exp_df['id'].values[-1]


class DatabaseManager:
    def __init__(self, local_db_config=None, checkpoint_dir=None):
        self.checkpoint_dir = checkpoint_dir
        # Local database uses in-memory db, if going to submit job to ComputeCanada to avoid huge I/O load.
        if 'database' not in local_db_config or local_db_config['database'] is None:
            # In-memory database
            local_db_config['database'] = None
        else:
            # Disk database
            local_db_config['database'] = os.path.join(self.checkpoint_dir, local_db_config['database'])
        self.local_db_config = local_db_config

        self._init_local_db_connection()
        self.episode_exp_start_id = self.local_db_exp_table_op.last_experience_id

    def _init_local_db_connection(self):
        # Init database URL
        if self.local_db_config is not None:
            self.local_db_url = sqla.engine.URL.create(**self.local_db_config)
        else:
            self.local_db_url = None
            raise ValueError("Please provide local_db_config!")


        # Create engine
        self.local_db_engine = sqla.create_engine(self.local_db_url) if self.local_db_url is not None else None

        # Create db tables if not exist
        if self.local_db_engine is not None:
            Base.metadata.create_all(self.local_db_engine)

        # Init local database table operators
        if self.local_db_engine is not None:
            self.local_db_session_maker = sessionmaker(bind=self.local_db_engine)
            self.local_db_session = self.local_db_session_maker()

            self.local_db_exp_table_op = ExperienceTableOperator(self.local_db_session)

        self.local_db_meta = sqla.MetaData()
        self.local_db_meta.reflect(bind=self.local_db_engine)


    ##########################################################################################################################
    #                                     ExperienceTable related operations                                                 #
    ##########################################################################################################################
    @property
    def collected_exp_num(self):
        return self.local_db_exp_table_op.last_experience_id

    def store_experience(self, obs, act, obs2, pb_rew, hc_rew, done, behavior_mode=None, obs_time=None, act_time=None, obs2_time=None):
        self.local_db_exp_table_op.store(obs, act, obs2, pb_rew, hc_rew, done, behavior_mode, obs_time, act_time, obs2_time)

    def sample_exp_batch(self, batch_size=64, device=None, reward_mem_len=None, agent_mem_len=None, multistep_size=None):
        return self.local_db_exp_table_op.sample_batch(batch_size, device, reward_mem_len, agent_mem_len, multistep_size)

    def get_latest_experience_id(self):
        latest_experience_id = self.local_db_exp_table_op.last_experience_id
        if latest_experience_id is None:
            latest_experience_id = 0
        return latest_experience_id

    def retrieve_last_experience_episode(self):
        """
        This function is used to retrieve last experience episode, which will be used to sample segment from online experiences.
        Note: only call this function after an episode, i.e., either reaching done or maximum_episode_length.
        """
        last_episode, self.episode_exp_start_id = self.local_db_exp_table_op.retrieve_last_experience_episode(self.episode_exp_start_id)
        return last_episode

    ##########################################################################################################################
    #                                       Other general purpose operations                                                 #
    ##########################################################################################################################
    def commit(self):
        self.local_db_session.commit()
        # self.cloud_db_session.commit()

    def _dump_mem_db_to_disk_db(self, db_disk_file):
        db_disk_conn = sqlite3.connect(db_disk_file)
        with db_disk_conn:
            for line in self.local_db_engine.raw_connection().iterdump():
                if line not in ('BEGIN;', 'COMMIT;'):
                    db_disk_conn.execute(line)
        db_disk_conn.commit()   # Commit
        db_disk_conn.close()    # Close the database connection

    def save_mem_checkpoint(self, time_step):
        if self.local_db_config['database'] is None:
            # In-memory database: dump it to disk."
            disk_pref_db_file = os.path.join(self.checkpoint_dir, 'Step-{}_Checkpoint_DB.sqlite3'.format(time_step))
            self._dump_mem_db_to_disk_db(disk_pref_db_file)
            # Rename the file to verify the completion of the saving in case of midway cutoff.
            verified_disk_pref_db_file = os.path.join(self.checkpoint_dir, 'Step-{}_Checkpoint_DB_verified.sqlite3'.format(time_step))
            os.rename(disk_pref_db_file, verified_disk_pref_db_file)
        else:
            # Disk database
            # Rename the file to verify the completion of the saving in case of midway cutoff.
            verified_disk_pref_db_file = os.path.join(self.checkpoint_dir,
                                                      'Step-{}_Checkpoint_DB_verified.sqlite3'.format(time_step))
            disk_pref_db_file = None
            for f_name in os.listdir(self.checkpoint_dir):
                if 'Checkpoint_DB' in f_name:
                    disk_pref_db_file = os.path.join(self.checkpoint_dir, f_name)
                    break

            if disk_pref_db_file is not None:
                # Close the database connection before renaming the file
                if self.local_db_engine:
                    self.local_db_session.close()
                    self.local_db_engine.dispose()
                os.rename(disk_pref_db_file, verified_disk_pref_db_file)
                # Reconnect the database and recreate aggregate
                self.local_db_config['database'] = verified_disk_pref_db_file
                self._init_local_db_connection()
            else:
                raise ValueError("disk_pref_db_file is None!")

    def restore_mem_checkpoint(self, time_step):
        # restore memory checkpoint
        if time_step == 0:
            disk_db_file = os.path.join(self.checkpoint_dir, 'Step-{}_Checkpoint_DB.sqlite3'.format(time_step))
        else:
            disk_db_file = os.path.join(self.checkpoint_dir, 'Step-{}_Checkpoint_DB_verified.sqlite3'.format(time_step))

        #
        if self.local_db_config['database'] is None:
            # Restore In-memory database
            self.local_db_config['database'] = None
            self._init_local_db_connection()

            # Connect to disk database and backup it to in-memory database
            disk_db_conn = sqlite3.connect(disk_db_file)
            disk_db_conn.backup(self.local_db_engine.raw_connection().connection)

            # Init local db connection again, because for the first initialization an empty database is created and used to initialize some variables.
            # Create db tables if not exist
            if self.local_db_engine is not None:
                Base.metadata.create_all(self.local_db_engine)

            # Init local database table operators
            if self.local_db_engine is not None:
                self.local_db_session_maker = sessionmaker(bind=self.local_db_engine)
                self.local_db_session = self.local_db_session_maker()

                self.local_db_exp_table_op = ExperienceTableOperator(self.local_db_session)

            self.local_db_meta = sqla.MetaData()
            self.local_db_meta.reflect(bind=self.local_db_engine)
        else:
            # Restore Disk database
            self.local_db_config['database'] = disk_db_file
            self._init_local_db_connection()

        # TODO: delete experiences after time_step to align precisely.
        print('Successfully restored database with {} experiences!'.format(self.collected_exp_num))


if __name__ == '__main__':

    local_db_config = {"drivername": "sqlite", "username": None, "password": None,
                       "database": "Step-0_Checkpoint_DB.sqlite3", "host": None, "port": None}
    local_db_config = {"drivername": "sqlite"}

    cloud_db_config = {"drivername": "postgresql", "username": "postgres", "password": "mlhmlh",
                       "database": "postgres", "host": "127.0.0.1", "port": "54321"}
    db_manager = DatabaseManager(local_db_config=local_db_config, cloud_db_config=cloud_db_config, checkpoint_dir='./')

    # db_manager.save_mem_checkpoint(1)

    test_local_db = True #False #True

    # Add experiences
    # db_manager.store_experience()
    obs_dim = 726
    act_dim = 16
    exp_num = 1000
    # # "2021-10-20-13-09-48_2021-10-20-13-10-18_NorthRiverCamera_clip"
    # video_start_time = datetime.datetime.strptime("2021-10-20-13-09-48", '%Y-%m-%d-%H-%M-%S')
    # video_end_time = datetime.datetime.strptime("2021-10-20-13-10-18", '%Y-%m-%d-%H-%M-%S')
    # for exp_i in range(exp_num):
    #     obs = np.random.rand(obs_dim)
    #     act = np.random.rand(act_dim)
    #     obs2 = np.random.rand(obs_dim)
    #     pb_rew = np.random.rand()
    #     hc_rew = np.random.rand()
    #     done = False
    #     behavior_mode = 'test'
    #     obs_time = video_start_time + exp_i*datetime.timedelta(seconds=2)
    #     act_time = video_start_time + exp_i*datetime.timedelta(seconds=3)
    #     obs2_time = video_start_time + exp_i*datetime.timedelta(seconds=2)
    #     if test_local_db:
    #         db_manager.local_db_exp_table_op.store(obs=obs, act=act, obs2=obs2, pb_rew=pb_rew, hc_rew=hc_rew, done=int(done), behavior_mode=behavior_mode,
    #                                                obs_time=obs_time, act_time=act_time, obs2_time=obs2_time)
    #     else:
    #         db_manager.cloud_db_exp_table_op.store(obs=obs, act=act, obs2=obs2, pb_rew=pb_rew, hc_rew=hc_rew, done=int(done), behavior_mode=behavior_mode,
    #                                                obs_time=obs_time, act_time=act_time, obs2_time=obs2_time)
    # db_manager.commit()
    # db_manager.local_db_exp_table_op.sample_batch(mem_len=16)
    # # db_manager.cloud_db_exp_table_op.sample_batch(mem_len=16)
    # # Add segments
    # seg_num = 100
    # seg_len = 15
    # seg_start_id = np.random.randint(1, exp_num - seg_len + 1, size=seg_num)
    # seg_end_id = seg_start_id + seg_len - 1
    # for seg_i in range(seg_num):
    #     if test_local_db:
    #         db_manager.local_db_seg_table_op.store(seg_start_id[seg_i - 1], seg_end_id[seg_i - 1], behavior_mode, 'NorthRiver-Camera', None, None, None,
    #                                                add_seg_pair_distance=True, reward_comp='')
    #     else:
    #         db_manager.cloud_db_seg_table_op.store(seg_start_id[seg_i - 1], seg_end_id[seg_i - 1], behavior_mode, 'NorthRiver-Camera', None, None, None)

    # # db_manager.local_db_seg_table_op.sample_segment(2)
    # # Simulate preference
    # pref_num = 3000
    # for i in range(pref_num):
    #     if i % 100 == 0:
    #         print(i)
    #     seg_1_id = np.random.randint(1, seg_num+1)
    #     seg_2_id = np.random.randint(1, seg_num + 1)
    #     pref_choice = 'right_better'
    #     pref_label = 1
    #     if test_local_db:
    #         db_manager.local_db_pref_table_op.store(seg_1_id, seg_2_id, pref_choice, pref_label, time_spend_for_labeling=None, teacher_id=None,
    #                                                 train_set=True)
    #     else:
    #         db_manager.cloud_db_pref_table_op.store(seg_1_id, seg_2_id, pref_choice, pref_label, time_spend_for_labeling=None, teacher_id=None,
    #                                                 train_set=True)
    # db_manager.cloud_db_pref_table_op.db_session.commit()
    # if test_local_db:
    #     db_manager.local_db_pref_table_op.training_dataset
    # else:
    #     db_manager.cloud_db_pref_table_op.training_dataset

    # # Test synchronization functions
    # db_manager.one_way_sync_cloud2local()

    # db_manager.local_db_seg_pair_dist_table_op.exist_segment_ids
    #
    # db_manager.save_mem_checkpoint(100)
    db_manager.restore_mem_checkpoint(100)

    # db_manager.one_way_sync_preference_table_cloud2local()
    # db_manager.one_way_sync_segment_table_local2cloud()