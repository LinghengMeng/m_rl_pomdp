import os
import pkg_resources
import sqlalchemy as db

# LAS related configuration
las_config = {"device_locator_csv": pkg_resources.resource_filename('m_rl', 'rsrc/device_locator_csv/Meander_AUG15.csv')}

# Communication manager related configuration
comm_manager_config = {
    "LAS_ML_Agent": {
        "IP": "127.0.0.1",
        "Obs_Port": 4010, "Act_Confirm_Port": 4020, "Config_Port": 4030, "Excitor_Raw_Act_Port": 4040,
        "potential_observation_sources": {
            "Uni_Sim": {
                "Actuators": {
                    "Map_Address": "/Uni_Sim/Observation/Actuators/{node_id}",
                    "Device": {"SM": {"size": 1, "min": 0, "max": 1},
                               "MO": {"size": 1, "min": 0, "max": 1},
                               "RS": {"size": 1, "min": 0, "max": 1},
                               "DR": {"size": 1, "min": 0, "max": 1},
                               "PC": {"size": 1, "min": 0, "max": 1}}},
                "Sensors": {
                    "Map_Address": "/Uni_Sim/Observation/Sensors/{node_id}",
                    "Device": {"IR": {"size": 1, "min": 0, "max": 750},
                               "GE": {"size": 64, "min": 24, "max": 36.5},
                               "SD": {"size": 1, "min": 0, "max": 1024}}},
            },
            "Pro_Sim": {
                "Actuators": {
                    "Map_Address": "/Pro_Sim/Observation/Actuators/{node_id}",
                    "Device": {"SM": {"size": 1, "min": 0, "max": 1},
                               "MO": {"size": 1, "min": 0, "max": 1},
                               "RS": {"size": 1, "min": 0, "max": 1},
                               "DR": {"size": 1, "min": 0, "max": 1},
                               "PC": {"size": 1, "min": 0, "max": 1}}},
                "Sensors": {
                    "Map_Address": "/Pro_Sim/Observation/Sensors/{node_id}",
                    "Device": {"IR": {"size": 1, "min": 0, "max": 750},
                               "GE": {"size": 5, "min": 0, "max": 2},
                               "SD": {"size": 1, "min": 0, "max": 1024}}},
            }
        },
        "potential_raw_actuator_control_destination": {'Uni_Sim'},
        "Map_Address": {
            "Sensor_Obs": {
                "IR": {
                    "LAS_Uni_Sim": {"Address_Paradigm": "/Sensor_Obs/Uni_Sim/IR_PT_SAMPLING_CONTROL/{node_id}",
                                    "Value_Format": {"Type": "string", "Size": 1, "Value_Range": [0, 750]}},
                    "LAS_Pro_Sim": {}},
                "GE": {
                    "LAS_Uni_Sim": {"Address_Paradigm": "/Sensor_Obs/Uni_Sim/GE_RAW/{node_id}",
                                    "Value_Format": {"Type": "string", "Size": 64, "Value_Range": [0, 50]}},
                    "LAS_Pro_Sim": {}},
                "SD": {
                    "LAS_Uni_Sim": {"Address_Paradigm": "/Uni_Sim/SD_PT_SAMPLING_CONTROL/{node_id}",
                                    "Value_Format": {"Type": "string", "Size": 1, "Value_Range": [0, 1024]}},
                    "LAS_Pro_Sim": {}}},
            "Actuator_Obs": {
                "LAS_Uni_Sim_Forward": {
                    "Address_Paradigm": "/Actuator_Obs/Uni_Sim/NODE/RAW_OUTPUT/{node_id}",
                    "Value_Format": {"Type": "string", "Format": "PIN float PIN float", "Value_Range": [0, 1]}},
                "LAS_Uni_Sim": {
                    "Address_Paradigm": "/Actuator_Obs/Uni_Sim/{actuator_type}_RAW_OUTPUT/{node_id}/{num}/{top_pin}_{bottom_pin}",
                    "Value_Format": {"Type": "string", "Format": "value", "Value_Range": [0, 1]}},
                "LAS_Pro_Sim": {}},
            "Actuator_Confirmation": {
               "LAS_Uni_Sim": {
                   "Address_Paradigm": "/Action_Confirm/Uni_Sim/NODE/{node_id}",
                   "Value_Format": {"Type": "string", "Format": "PIN 1 PIN 1", "Value_Range": [0, 1]},
               },
               "LAS_Pro_Sim": {}},
            "Parameterized_Action_Confirmation": {
                "ParamAction": {
                    "Address_Paradigm": "/Para_Action_Confirmation/LAS_ML_Agent/{Source_Server}/PA_Confirmation",
                    "Source_Server": {
                      "LAS_Uni_Sim": {"Value": "received"},
                      "LAS_Pro_Sim": { }}}}
        }
    },
    "LAS_Unity_Sim": {
        "IP": "127.0.0.1", "Port": 7000,  # (Note: 7000 is not used)
        "Excitor_Port": 7010, "Video_Capture_Port": 7020,
        "Map_Address": {
            "Actuator_Command": {
              "SM": "/LAS_Unity_Sim/{Source_Server}/SM_Execution_Command/{node_id}/{num_id}",
              "MO": "/LAS_Unity_Sim/{Source_Server}/MO_Execution_Command/{node_id}/{num_id}",
              "RS": "/LAS_Unity_Sim/{Source_Server}/RS_Execution_Command/{node_id}/{num_id}",
              "DR": "/LAS_Unity_Sim/{Source_Server}/DR_Execution_Command/{node_id}/{num_id}",
              "PC": "/LAS_Unity_Sim/{Source_Server}/PC_Execution_Command/{node_id}/{num_id}"
            },
            "Node_Actuator_Obs": {
              "RAW_Actuators": "/NODE/RAW_OUTPUT/{node_id}"
            }
        }
    },
    "LAS_Processing_Sim": {
        "IP": "127.0.0.1", "Port": 4000,
        "Map_Address": {
            "Sensor_Sim": {
                "IR": "/NODE/IR_PT_SAMPLING_CONTROL/{node_id}",
                "GE": "/NODE/GE_PT_SAMPLING_CONTROL/{node_id}",
                "SD": "/NODE/SD_PT_SAMPLING_CONTROL/{node_id}"},
            "Actuator_Command": {},
            "Parameterized_Action_Value": {
                "ParamAction" : "/LAS_Processing_Sim/{source_server}/PA_Value"}
        }
    },
    # LAS_GUI_OSC_Server has functions to send and receive parameters and status from Processing_Sim
    "LAS_GUI_OSC_Server": {
        "IP": "127.0.0.1", "Port": 3006,
        "MAP_Address": {
            "Receive_OSC": {
                "Set_Dat_Parameter": "/setDatParameter"  # value: behaviour_name parameter_name parameter_value
            }
        }
    }
}

# Internal environment related configuration
internal_env_config = {
    "max_episode_steps": 100,               # maximum episode steps
    "wait_time_for_action_execution": 0,    # Wait for executing action and start receiving sensors and actuators state
    # Action space configuration
    "action_space": {
        # choose raw_action_space or para_action_space
        "employed_action_space": "para_action_space",  # "raw_action_space"
        # valid action value range for learning algorithm
        "act_value": {"act_val_max": 1, "act_val_min": -1},
        # 1. Raw action space configuration
        "raw_act_space": {
            "LAS_for_action_execution": "Uni_Sim",  # execute raw action on Uni_Sim or Pro_Sim
            "device_group": ["NR"],
            "actuator_type": ["SM", "MO", "RS", "DR", "PC"],
            "actuator_val": [0, 1]
        },
        # 2. Parameterized action space configuration
        #    Important: check data_behaviours_2.html in Gaslight-OSC-Server to confirm the value range and data type of each parameter.
        "para_act_space": {
            # GridRunner related parameters
            # "GridRunner": {
            #     "gridScale": {"type": "float", "min": 0.5, "max": 5},
            #     "nParticles": {"type": "int", "min": 5, "max": 2000}
            # },
            # GridRunner in SR
            # "GridRunner/Source_523": {
            #     "active": {"type": "bool", "min": False, "max": True},
            #     "sourceRotation": {"type": "int", "min": 0, "max": 2},
            #     "spread": {"type": "float", "min": 0, "max": 6.2832},
            #     "heading": {"type": "float", "min": 0, "max": 6.2832},
            #     "burstFreq": {"type": "int", "min": 10, "max": 5000},
            #     "burstQty": {"type": "int", "min": 1, "max": 250},
            #     "yvelocity": {"type": "float", "min": 0, "max": 1},
            #     "influenceSize": {"type": "int", "min": 0, "max": 1500},
            #     "influenceIntensity": {"type": "float", "min": 0, "max": 1},
            #     "maxspeed": {"type": "float", "min": 0.5, "max": 10}
            # },
            # "GridRunner/Source_522": {
            #     "active": {"type": "bool", "min": False, "max": True},
            #     "sourceRotation": {"type": "int", "min": 0, "max": 2},
            #     "spread": {"type": "float", "min": 0, "max": 6.2832},
            #     "heading": {"type": "float", "min": 0, "max": 6.2832},
            #     "burstFreq": {"type": "int", "min": 10, "max": 5000},
            #     "burstQty": {"type": "int", "min": 1, "max": 250},
            #     "yvelocity": {"type": "float", "min": 0, "max": 1},
            #     "influenceSize": {"type": "int", "min": 0, "max": 1500},
            #     "influenceIntensity": {"type": "float", "min": 0, "max": 1},
            #     "maxspeed": {"type": "float", "min": 0.5, "max": 10}
            # },
            # "GridRunner/Source_520": {
            #     "active": {"type": "bool", "min": False, "max": True},
            #     "sourceRotation": {"type": "int", "min": 0, "max": 2},
            #     "spread": {"type": "float", "min": 0, "max": 6.2832},
            #     "heading": {"type": "float", "min": 0, "max": 6.2832},
            #     "burstFreq": {"type": "int", "min": 10, "max": 5000},
            #     "burstQty": {"type": "int", "min": 1, "max": 250},
            #     "yvelocity": {"type": "float", "min": 0, "max": 1},
            #     "influenceSize": {"type": "int", "min": 0, "max": 1500},
            #     "influenceIntensity": {"type": "float", "min": 0, "max": 1},
            #     "maxspeed": {"type": "float", "min": 0.5, "max": 10}
            # },
            # GridRunner in NR
            # "GridRunner/Source_530": {
            #     "active": {"type": "bool", "min": False, "max": True},
            #     "sourceRotation": {"type": "int", "min": 0, "max": 2},
            #     "spread": {"type": "float", "min": 0, "max": 6.2832},
            #     "heading": {"type": "float", "min": 0, "max": 6.2832},
            #     "burstFreq": {"type": "int", "min": 10, "max": 5000},
            #     "burstQty": {"type": "int", "min": 1, "max": 250},
            #     "yvelocity": {"type": "float", "min": 0, "max": 1},
            #     "influenceSize": {"type": "int", "min": 0, "max": 1500},
            #     "influenceIntensity": {"type": "float", "min": 0, "max": 1},
            #     "maxspeed": {"type": "float", "min": 0.5, "max": 10}
            # },
            # "GridRunner/Source_528": {
            #     "active": {"type": "bool", "min": False, "max": True},
            #     "sourceRotation": {"type": "int", "min": 0, "max": 2},
            #     "spread": {"type": "float", "min": 0, "max": 6.2832},
            #     "heading": {"type": "float", "min": 0, "max": 6.2832},
            #     "burstFreq": {"type": "int", "min": 10, "max": 5000},
            #     "burstQty": {"type": "int", "min": 1, "max": 250},
            #     "yvelocity": {"type": "float", "min": 0, "max": 1},
            #     "influenceSize": {"type": "int", "min": 0, "max": 1500},
            #     "influenceIntensity": {"type": "float", "min": 0, "max": 1},
            #     "maxspeed": {"type": "float", "min": 0.5, "max": 10}
            # },
            # "GridRunner/Source_525": {
            #     "active": {"type": "bool", "min": False, "max": True},
            #     "sourceRotation": {"type": "int", "min": 0, "max": 2},
            #     "spread": {"type": "float", "min": 0, "max": 6.2832},
            #     "heading": {"type": "float", "min": 0, "max": 6.2832},
            #     "burstFreq": {"type": "int", "min": 10, "max": 5000},
            #     "burstQty": {"type": "int", "min": 1, "max": 250},
            #     "yvelocity": {"type": "float", "min": 0, "max": 1},
            #     "influenceSize": {"type": "int", "min": 0, "max": 1500},
            #     "influenceIntensity": {"type": "float", "min": 0, "max": 1},
            #     "maxspeed": {"type": "float", "min": 0.5, "max": 10}
            # },
            # AmbientWaves related parameters: both North River and South River share the same set of parameters.l
            # "AmbientWaves": {
            #     "waveActive": {"type": "bool", "min": False, "max": True},
            #     "velocity": {"type": "float", "min": 0, "max": 2},
            #     "period": {"type": "float", "min": 0, "max": 1},
            #     "angle": {"type": "float", "min": 0, "max": 6.283},
            #     "amplitude": {"type": "float", "min": 0, "max": 1},
            # },
            # "AmbientWaves/Wave_SR_1": {
            #     "waveActive": {"type": "bool", "min": False, "max": True},
            #     "velocity": {"type": "float", "min": 0, "max": 2},
            #     "period": {"type": "float", "min": 0, "max": 1},
            #     "angle": {"type": "float", "min": 0, "max": 6.283},
            #     "amplitude": {"type": "float", "min": 0, "max": 1},
            # },
            # "AmbientWaves/Wave_NR_3": {
            #     "waveActive": {"type": "bool", "min": False, "max": True},
            #     "velocity": {"type": "float", "min": 0, "max": 2},
            #     "period": {"type": "float", "min": 0, "max": 1},
            #     "angle": {"type": "float", "min": 0, "max": 6.283},
            #     "amplitude": {"type": "float", "min": 0, "max": 1},
            # },
            # Excitors related parameters
            "Excitors": {
                "size": {"type": "int", "min": 40, "max": 2000},
                "coreSize": {"type": "float", "min": 0, "max": 1},
                "lifespan": {"type": "int", "min": 500, "max": 20000},
                "masterIntensity": {"type": "float", "min": 0, "max": 1},
                "excitorSpeedLimit": {"type": "float", "min": 0, "max": 1},
                "attractorAngleSpeed": {"type": "float", "min": 0, "max": 0.25},
                "forceScalar": {"type": "float", "min": 0, "max": 5},
                "bgHowOften": {"type": "int", "min": 250, "max": 10000},
                "maxExcitorAmount": {"type": "int", "min": 1, "max": 35}
            },
            # ElectricCells related parameters
            # "ElectricCells": {}
        },
    },
    # Observation space configuration
    #   obs_construction_method in ["concatenate", "average"]
    "time_window_for_obs_collection": 1,  # 2 # TODO: need to be tuned (ceil(time_window_for_obs_collection*obs_frequency))
    "obs_space": {"proprioception": {
                                     # "device_group": ["NR"],
                                     "device_group": ["NR", "SR", "MG", "TG"],
                                     # "device_group": ["NR", "SR"],
                                     # "device_group": ["NR"],
                                     "actuator_type": {"SM": {"size": 1, "min": 0, "max": 1},
                                                       "MO": {"size": 1, "min": 0, "max": 1},
                                                       "RS": {"size": 1, "min": 0, "max": 1},
                                                       "DR": {"size": 1, "min": 0, "max": 1},  # DR is treated as two separate acutoators corresponding to both top and bottom pin.
                                                       "PC": {"size": 1, "min": 0, "max": 1}},
                                     # "actuator_type": {"MO": {"size": 1, "min": 0, "max": 1},
                                     #                   "RS": {"size": 1, "min": 0, "max": 1}},
                                     "obs_frequency": 1,
                                     "obs_construction_method": "concatenate",  # "concatenate", "average"
                                     "obs_source_server": 'Pro_Sim'         # 'Pro_Sim', 'Uni_Sim'
                                     },
                  "exteroception": {# "device_group": ["NR"],
                                    # "device_group": [],
                                    "device_group": ["NR", "SR", "MG", "TG"],
                                    "sensor_type": {"IR": {"size": 1, "min": 0, "max": 1},
                                                    "GE": {"size": 5, "min": 0, "max": 1},
                                                    "SD": {"size": 1, "min": 0, "max": 1}},
                                    "obs_frequency": 1,
                                    "obs_construction_method": "concatenate",# "concatenate", "average"
                                    "obs_source_server": 'Pro_Sim'},
                  "add_velocity_to_obs": False,
                  "add_past_obs_to_obs": False,
                  "add_act_to_obs": False},
    "reward_function": {
        # learned or handcrafted
        #    1. "handcrafted_reward_component", 2. "mlp_reward_component", 3. "lstm_reward_component"
        "reward_type": "handcrafted_reward_component",
        "handcrafted_reward_type": "active_all",   # active_all, calm_all, active_NR_calm_SR, active_SR_calm_NR
        # "reward_range_0_pos_1", "reward_range_neg_1_pos_1", "reward_range_0_pos_2", "reward_range_neg_2_pos_2", "reward_range_0_pos_10","reward_range_0_pos_100"
        "hc_reward_range": "reward_range_0_pos_1"  #
    }
}

# Database related configuration
db_config = {
    # Google_Cloud_DB is to configurate Google cloud databased when running the algorithm in real application.
    "Google_Cloud_DB": {"postgresql": {"drivername": "postgresql", "username": "postgres", "password": "mlhmlh",
                                       "database": "postgres", "host": "127.0.0.1", "port": "54321"}},
    # Local_DB is to configurate local databased used for testing purpose.
    "Local_Disk_DB": {"drivername": "sqlite", "username": None, "password": None,
                       "database": "Step-0_Checkpoint_DB.sqlite3", "host": None, "port": None},
    "Local_In_Memory_DB": {"drivername": "sqlite"}
}



