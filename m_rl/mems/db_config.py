
# Database table configurations
db_table_config = {"experience_table": {"id": {"data_type": "int", "default": None, "primary_key": True, "foreign_key": None},
                                        "obs": {"data_type": "array", "default": None, "primary_key": None, "foreign_key": None},
                                        "act": {"data_type": "array", "default": None, "primary_key": None, "foreign_key": None},
                                        "obs2": {"data_type": "array", "default": None, "primary_key": None, "foreign_key": None},
                                        "pb_rew": {"data_type": "float", "default": None, "primary_key": None, "foreign_key": None},
                                        "hc_rew": {"data_type": "float", "default": None, "primary_key": None, "foreign_key": None},
                                        "done": {"data_type": "int", "default": None, "primary_key": None, "foreign_key": None},
                                        "sampled_num": {"data_type": "int", "default": 0, "primary_key": None, "foreign_key": None},
                                        "behavior_mode": {"data_type": "text", "default": None, "primary_key": None, "foreign_key": None},
                                        # Time related columns are used to find the correspondence between experiences and video clip.
                                        "obs_time": {"data_type": "time", "default": None, "primary_key": None, "foreign_key": None},
                                        "act_time": {"data_type": "time", "default": None, "primary_key": None, "foreign_key": None},
                                        "obs2_time": {"data_type": "time", "default": None, "primary_key": None, "foreign_key": None},
                                        "create_time": {"data_type": "time", "default": None, "primary_key": None, "foreign_key": None}}}