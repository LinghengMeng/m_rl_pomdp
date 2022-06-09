import os
import numpy as np
import pandas as pd
import json
import time
from datetime import datetime
import math
from multiprocessing import Queue
from threading import Thread
from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_server
from pythonosc import udp_client
import gym

import m_rl.las_config as config
import torch


class CommunicationManager(object):
    """
    Class used for managing the communication between ML-Agent and external clients i.e. Unity Simulator and
    Processing Simulator.
    """
    def __init__(self, comm_manager_config, las_config=config.las_config):
        """"""
        self.config = comm_manager_config   # communication configuration
        self.las_config = las_config        # las device information
        # Read device location.csv
        self.device_locator = pd.read_csv(self.las_config['device_locator_csv'])
        ###############################################################
        #           Initialize sensor and actuator dictionary         #
        ###############################################################
        # device_dict: (a dict of lists of device identification with the format of {device_node_id}_{device_type}_{device_num}_{device_uid})
        #   "sensors":
        #       "IR": ["{device_node_id}_{device_type}_{device_num}_{device_uid}"], "GE": [], "SD": []
        #   "actuators":
        #       "SM": [], "MO": [], "RS": [], "DR": [], "PC": []
        # device_map_dict: (a dict of dict of string of device identification, used to find the device by node_id and pin)
        #   "sensors":
        #       "device_node_id":
        #           "device_uid": "{device_node_id}_{device_type}_{device_num}_{device_uid}"
        #   "actuators":
        #       "device_node_id":
        #           "device_uid": "{device_node_id}_{device_type}_{device_num}_{device_uid}"
        self.device_dict = {"sensors": {}, "actuators": {}}
        self.device_map_dict = {"sensors": {}, "actuators": {}}    # Used to find the device type by node_id and pin

        self.actuator_dict = {}
        for index, row in self.device_locator.iterrows():
            device_type = row['DEVICE']
            device_node_id = str(row['NODE ID'])
            device_num = row['NUM']
            device_uid = str(row['UID'])
            device_config = row['CONFIG']

            if device_num == "--":
                continue
            # sensors
            if device_type in ["IR", "GE", "SD"]:
                # Initialize device dict
                if device_type not in self.device_dict["sensors"]:
                    self.device_dict["sensors"][device_type] = []
                self.device_dict["sensors"][device_type].append("{}_{}_{}_{}".format(device_node_id, device_type, device_num, device_uid))

                # initialize device_map_dict
                if device_node_id not in self.device_map_dict["sensors"]:
                    self.device_map_dict["sensors"][device_node_id] = {}
                # device_node_id + device_uid corresponds to a unique device
                self.device_map_dict["sensors"][device_node_id][device_uid] = "{}_{}_{}_{}".format(device_type, device_node_id, device_num, device_uid)

            # actuators
            elif device_type in ["SM", "MO", "RS", "DR", "PC"]:
                if device_type not in self.actuator_dict:
                    self.actuator_dict[device_type] = []
                    self.device_dict["actuators"][device_type] = []
                # Initialize device dict
                self.actuator_dict[device_type].append("{}_{}_{}_{}".format(device_node_id, device_type, device_num, device_uid))
                self.device_dict["actuators"][device_type].append("{}_{}_{}_{}".format(device_node_id, device_type, device_num, device_uid))
                # Bottom pin for Double Rebel Star
                if type(device_config) == str:
                    device_bottom_pin = device_config.split(' ')[1].split(';')[0]
                    self.actuator_dict[device_type].append("{}_{}_{}_{}".format(device_node_id, device_type, device_num, device_bottom_pin))
                    self.device_dict["actuators"][device_type].append("{}_{}_{}_{}".format(device_node_id, device_type, device_num, device_bottom_pin))

                # Initialize device_map_dict
                if device_node_id not in self.device_map_dict["actuators"]:
                    self.device_map_dict["actuators"][device_node_id] = {}
                self.device_map_dict["actuators"][device_node_id][device_uid] = "{}_{}_{}_{}".format(device_type, device_node_id,
                                                                                                     device_num, device_uid)
                if type(device_config) == str:
                    device_bottom_pin = device_config.split(' ')[1].split(';')[0]
                    self.device_map_dict["actuators"][device_node_id][device_bottom_pin] = "{}_{}_{}_{}".format(device_type, device_node_id,
                                                                                                                device_num, device_bottom_pin)
        ###############################################################
        #                     Initialize buffers                      #
        #                                                             #
        ###############################################################
        max_buffer_size = 50
        # Sensor and Actuator Observation Buffer and Actuator Confirmation Buffer
        self.sensor_obs_buffer = {}
        self.actuator_obs_buffer = {}
        self.actuator_confirm_buffer = {}
        self.parameterized_action_conf_buffer = {}
        self.excitor_raw_act_buffer = {}            # buffer stores raw action elicited by excitor
        for source_server in self.config['LAS_ML_Agent']['potential_observation_sources']:
            # Sensor Obs
            self.sensor_obs_buffer[source_server] = {}
            for device_type in self.device_dict["sensors"]:
                self.sensor_obs_buffer[source_server][device_type] = {}
                for device_id in self.device_dict["sensors"][device_type]:
                    self.sensor_obs_buffer[source_server][device_type][device_id] = Queue(maxsize=max_buffer_size)
            # Actuator Obs and Confirmation
            self.actuator_obs_buffer[source_server] = {}
            self.actuator_confirm_buffer[source_server] = {}    # Used only when control raw actuators
            self.excitor_raw_act_buffer[source_server] = {}     #
            for device_type in self.actuator_dict:
                self.actuator_obs_buffer[source_server][device_type] = {}
                self.actuator_confirm_buffer[source_server][device_type] = {}
                self.excitor_raw_act_buffer[source_server][device_type] = {}
                for device_id in self.actuator_dict[device_type]:
                    self.actuator_obs_buffer[source_server][device_type][device_id] = Queue(maxsize=max_buffer_size)
                    self.actuator_confirm_buffer[source_server][device_type][device_id] = Queue(maxsize=max_buffer_size)
                    self.excitor_raw_act_buffer[source_server][device_type][device_id] = Queue(maxsize=max_buffer_size)
            # Parameterized Action Confirmation
            self.parameterized_action_conf_buffer[source_server] = Queue(maxsize=max_buffer_size)

        ###############################################################
        #                 Create server and clients                   #
        ###############################################################
        self.obs_server, self.act_confirm_server, self.config_server, self.excitor_raw_act_server = self._create_server()
        self.unity_client, self.processing_client, self.gui_osc_server_client = self._create_client()

        self.video_capture_client = self._create_video_capture_client()
        self.excitor_client = self._create_excitor_client()
        self.node_client = self._create_node_client_for_raw_act_execution()

        # Find the device observation value range by source_server and device_name
        self.device_obs_value_range = {}
        for source_server in self.config['LAS_ML_Agent']['potential_observation_sources']:
            self.device_obs_value_range[source_server] = {}
            for device_type in ["Actuators", "Sensors"]:
                for device_name in self.config['LAS_ML_Agent']['potential_observation_sources'][source_server][device_type]["Device"]:
                    device_val_min = self.config['LAS_ML_Agent']['potential_observation_sources'][source_server][device_type]["Device"][device_name]["min"]
                    device_val_max = self.config['LAS_ML_Agent']['potential_observation_sources'][source_server][device_type]["Device"][device_name]["max"]
                    self.device_obs_value_range[source_server][device_name] = {"min": device_val_min, "max": device_val_max}

        self.ir_min = 0
        self.ir_max = 750
        self.ge_min = 0
        self.ge_max = 50
        self.sd_min = 0
        self.sd_max = 1024

    def empty_obs_buffer(self):
        """Clear all buffers"""
        # TODO: test efficiency of these two for loops
        for source_server in self.config['LAS_ML_Agent']['potential_observation_sources']:
            # Empty observation buffers for sensors
            for device_type in self.sensor_obs_buffer[source_server]:
                for device_id in self.sensor_obs_buffer[source_server][device_type]:
                    while not self.sensor_obs_buffer[source_server][device_type][device_id].empty():
                        self.sensor_obs_buffer[source_server][device_type][device_id].get()

            # Empty observation buffers for actuators
            for device_type in self.actuator_obs_buffer[source_server]:
                for device_id in self.actuator_obs_buffer[source_server][device_type]:
                    while not self.actuator_obs_buffer[source_server][device_type][device_id].empty():
                        self.actuator_obs_buffer[source_server][device_type][device_id].get()

    def empty_raw_act_confirm_buffer(self):
        """ """
        for source_server in self.config['LAS_ML_Agent']['potential_observation_sources']:
            for device_type in self.actuator_confirm_buffer[source_server]:
                for device_id in self.actuator_confirm_buffer[source_server][device_type]:
                    while not self.actuator_confirm_buffer[source_server][device_type][device_id].empty():
                        self.actuator_confirm_buffer[source_server][device_type][device_id].get()

    def empty_para_act_confirm_buffer(self):
        """ """

    def empty_excitor_raw_act_buffer(self):
        for source_server in self.config['LAS_ML_Agent']['potential_observation_sources']:
            for device_type in self.excitor_raw_act_buffer[source_server]:
                for device_id in self.excitor_raw_act_buffer[source_server][device_type]:
                    while not self.excitor_raw_act_buffer[source_server][device_type][device_id].empty():
                        self.excitor_raw_act_buffer[source_server][device_type][device_id].get()

    def empty_all_buffers(self):
        self.empty_obs_buffer()
        self.empty_raw_act_confirm_buffer()
        self.empty_para_act_confirm_buffer()

    ##############################################################################
    #****************************************************************************#
    #                              Server Functions                              #
    #****************************************************************************#
    ##############################################################################
    #####################################
    #       Message for Observation.    #
    #####################################
    # TODO: change the address format for sensors that used to form observations.
    def sensor_obs_from_uni_sim_handler(self, address, value):
        """
        Handle observation of sensors from Unity-Simulator.
        Example address format:
            "/Sensor_Obs/Uni_Sim/IR_PT_SAMPLING_CONTROL/{device_node_id}"
        :param address:
        :param value:
        :return:
        """
        import pdb; pdb.set_trace()
        _, _, source_server, message_indicator, device_node_id = address.split('/')
        device_type = message_indicator.split('_')[0]
        # print("{}: {}".format(address, value))

        # Pre-processing value
        if device_type == "IR":
            value = (float(value) - self.ir_min) / (self.ir_max - self.ir_min)
        elif device_type == "GE":
            value = [(float(x) - self.ge_min)/(self.ge_max - self.ge_min) for x in value.split(' ')]
        elif device_type == "SD":
            value = (float(value) - self.sd_min) / (self.sd_max - self.sd_min)
        else:
            raise ValueError("Sensor data from {}-{} is not properly handled.".format(device_type, device_node_id))

        # Throw the first one, if Queue is full.
        if self.sensor_obs_buffer[source_server][device_type][device_node_id].full():
            self.sensor_obs_buffer[source_server][device_type][device_node_id].get()
        # Put into the Queue
        self.sensor_obs_buffer[source_server][device_type][device_node_id].put(value)




    def individual_actuator_obs_from_uni_sim_handler(self, address, value):
        """
        Handle observation of actuators extracted from Unity-Simulator.
        Example address format:
            "/Actuator_Obs/Uni_Sim/{actuatorType}_RAW_OUTPUT/{actuatorNodeId}/{actuatorNum}/{pinNumber}_{bottomPinNumber}"
        where bottomPinNumber = -1 means bottomPin is not used.
        Example value format:
            "float" or "float float" if bottomPinNumber != -1
        :param address:
        :param value:
        :return:
        """
        _, _, source_server, message_indicator, device_node_id, device_num, device_pins = address.split('/')
        device_type = message_indicator.split('_')[0]
        device_pin, device_bottom_pin = device_pins.split('_')
        # if device_type == 'MO':
        #     print("{}: {}".format(address, value))
        # Convert value to float
        converted_value = [float(val) for val in value.split(' ')]
        if device_bottom_pin != "-1" and len(converted_value) != 2:
            raise ValueError("Actuator value and PINs do not match.")

        # Add actuator state corresponding to Pin
        device_id = "{}_{}_{}_{}".format(device_node_id, device_type, device_num, device_pin)
        if self.actuator_obs_buffer[source_server][device_type][device_id].full():
            self.actuator_obs_buffer[source_server][device_type][device_id].get()
        self.actuator_obs_buffer[source_server][device_type][device_id].put(converted_value[0])

        # Add actuator state corresponding to bottomPin
        if device_bottom_pin != "-1":
            device_id = "{}_{}_{}_{}".format(device_node_id, device_type, device_num, device_bottom_pin)
            if self.actuator_obs_buffer[source_server][device_type][device_id].full():
                self.actuator_obs_buffer[source_server][device_type][device_id].get()
            self.actuator_obs_buffer[source_server][device_type][device_id].put(converted_value[1])

    def sensor_obs_handler(self, address, *sensor_states):
        """
        Handle observation of sensors from Processing-Simulator and Unity-Simulator.
        (Note: The message is sent from Node.pdb for Pro_Sim and from Exteroceptor.cs for Uni_Sim. And
            Different from actuators, each node only has one sensor.)
        Example address format:
            "/{Pro_Sim or Uni_Sim}/Observation/Sensors/{device_node_id}"
            (PIN, float, float, ...) for Pro_Sim and "PIN float float ..." for Uni_Sim (PIN: int, float: range defined in las_config.py)
        :param address:
        :param sensor_states:
        :return:
        """
        # print("{}: {}".format(address, sensor_states))
        _, source_server, _, _, device_node_id = address.split('/')
        # OSC message from Uni_Sim is a string with sensor pin and reading, while the message from Pro_Sim is a
        # tuple with the value of sensor's pin followed by its reading.
        if source_server == "Uni_Sim":
            sensor_states = sensor_states[0].strip(' ').split(' ')
        uid = str(sensor_states[0])
        # Retrieve device information
        device_type, device_node_id, device_num, device_uid = self.device_map_dict["sensors"][device_node_id][uid].split('_')

        # Extract sensor value and convert to valid value range
        val_min = self.device_obs_value_range[source_server][device_type]["min"]
        val_max = self.device_obs_value_range[source_server][device_type]["max"]
        val = [(float(v) - val_min)/(val_max - val_min) for v in sensor_states[1:]]

        # Throw the first one, if Queue is full.
        device_id = "{}_{}_{}_{}".format(device_node_id, device_type, device_num, device_uid)
        if self.sensor_obs_buffer[source_server][device_type][device_id].full():
            self.sensor_obs_buffer[source_server][device_type][device_id].get()
        # Put into the Queue
        self.sensor_obs_buffer[source_server][device_type][device_id].put(val)

    def actuator_obs_handler(self, address, *actuator_states):
        """
        Handle observation of actuators extracted from Processing-Simulator and Unity-Simulator.
        (Note: The message is sent from Node.pdb for Pro_Sim and from Node.cs for Uni_Sim.)
        Example address and value format:
            "/{Pro_Sim or Uni_Sim}/Observation/Actuators/{node_id}"
            (PIN, float, PIN, float, ...) for Pro_Sim and "PIN float PIN float ..." for Uni_Sim (PIN: int, float: [0,1])
        :param address:
        :param actuator_states: value is a string of values of PIN and float value
        :return:
        """
        # print("{}: {}".format(address, actuator_states))
        _, source_server, _, _, device_node_id = address.split('/')
        # OSC message from Uni_Sim is a string with all actuators' states, while the message from Pro_Sim is a
        # tuple with the value of an actuator's pin followed by its state.
        if source_server == "Uni_Sim":
            actuator_states = actuator_states[0].strip(' ').split(' ')

        # Put actuator state to the corresponding buffer
        for i in range(len(actuator_states)):
            if i % 2 == 0:
                uid = str(actuator_states[i])         # uid works as a key to map to device info
                val = float(actuator_states[i + 1])   # all actuator state value is float
                # Retrieve device information
                device_type, device_node_id, device_num, device_uid = self.device_map_dict["actuators"][device_node_id][uid].split(
                    '_')
                # Throw the first one, if Queue is full.
                device_id = "{}_{}_{}_{}".format(device_node_id, device_type, device_num, device_uid)
                if self.actuator_obs_buffer[source_server][device_type][device_id].full():
                    self.actuator_obs_buffer[source_server][device_type][device_id].get()
                # Put into the Queue
                self.actuator_obs_buffer[source_server][device_type][device_id].put(val)

    def parameterized_action_obs_handler(self, address, value):
        """
        Handle observation of parameterized action.
        :param address:
        :param value:
        :return:
        """

    #####################################
    #    Message for Excitor Action.    #
    #####################################
    def excitor_raw_act_from_uni_sim_handler(self, address, value):
        """Handle raw action elicited by excitor. This is similar to raw actuator observation."""
        _, _, source_server, _, message_indicator, device_node_id = address.split('/')
        # print("{}: {}".format(address, value))
        actuator_state = value.strip(' ').split(' ')
        for i in range(len(actuator_state)):
            if i % 2 == 0:
                uid = actuator_state[i]
                val = float(actuator_state[i + 1])
                device_type, device_node_id, device_num, device_uid = self.device_map_dict["actuators"][device_node_id][uid].split(
                    '_')
                # Throw the first one, if Queue is full.
                device_id = "{}_{}_{}_{}".format(device_node_id, device_type, device_num, device_uid)
                if self.excitor_raw_act_buffer[source_server][device_type][device_id].full():
                    self.excitor_raw_act_buffer[source_server][device_type][device_id].get()
                # Put into the Queue
                self.excitor_raw_act_buffer[source_server][device_type][device_id].put(val)

    #####################################
    #        Message for Action.        #
    #####################################
    def actuator_confirm_from_uni_sim_handler(self, address, value):
        """
        Handle confirmation of receiving actuator execution.
        :param address:
        :param value:
        :return:
        """
        _, _, source_server, _, device_node_id = address.split('/')
        # print("{}: {}".format(address, value))
        action_confirm = value.split(' ')
        for i in range(len(action_confirm)):
            if i % 2 == 0:
                uid = action_confirm[i]
                val = float(action_confirm[i + 1])
                device_type, device_node_id, device_num, device_uid = self.device_map_dict["actuators"][device_node_id][uid].split(
                    '_')
                # Throw the first one, if Queue is full.
                device_id = "{}_{}_{}_{}".format(device_node_id, device_type, device_num, device_uid)
                if self.actuator_confirm_buffer[source_server][device_type][device_id].full():
                    self.actuator_confirm_buffer[source_server][device_type][device_id].get()
                # Put into the Queue
                self.actuator_confirm_buffer[source_server][device_type][device_id].put(val)

    def parameterized_action_confirmation_handler(self, address, value):
        """
        Handle confirmation of receiving parameterized action.
        :param address:
        :param value:
        :return:
        """
        _, _, _, source_server, message_indicator = address.split('/')
        if self.parameterized_action_conf_buffer[source_server].full():
            self.parameterized_action_conf_buffer[source_server].get()
        # Put into the Queue
        self.parameterized_action_conf_buffer[source_server].put(value)

    #####################################
    #    Message for Configuration.     #
    #####################################
    def configuration_confirmation_handler(self, address, value):
        """
        Handle confirmation of receiving configuration information.
        :param address:
        :param value:
        :return:
        """


    def config_handler(self, address, value):
        print("Receive configuration information.")
        self.config = json.loads(value)

    ##############################################################################
    #****************************************************************************#
    #                              Client Functions                              #
    #****************************************************************************#
    ##############################################################################


    ##############################################################################
    #                                  Utilities                                 #
    ##############################################################################
    def _create_server(self):
        server_ip = self.config['LAS_ML_Agent']['IP']
        obs_server_port = self.config['LAS_ML_Agent']['Obs_Port']
        act_confirm_server_port = self.config['LAS_ML_Agent']['Act_Confirm_Port']
        config_server_port = self.config['LAS_ML_Agent']['Config_Port']
        excitor_raw_act_server_port = self.config['LAS_ML_Agent']['Excitor_Raw_Act_Port']

        # Observation from either Uni_Sim or Pro_Sim.
        #   1st * corresponds to the source of observation
        #   2nd * corresponds to the node_id
        obs_dispatcher = Dispatcher()
        obs_dispatcher.map("/*/Observation/Sensors/*", self.sensor_obs_handler)
        obs_dispatcher.map("/*/Observation/Actuators/*", self.actuator_obs_handler)
        # obs_server = osc_server.ThreadingOSCUDPServer((server_ip, obs_server_port), obs_dispatcher)
        obs_server = osc_server.BlockingOSCUDPServer((server_ip, obs_server_port), obs_dispatcher)

        # Action Confirmation
        act_confirm_dispatcher = Dispatcher()
        act_confirm_dispatcher.map("/Actuation_Confirm/Uni_Sim/*", self.actuator_confirm_from_uni_sim_handler)
        # act_confirm_server = osc_server.ThreadingOSCUDPServer((server_ip, act_confirm_server_port),
        #                                                       act_confirm_dispatcher)
        act_confirm_server = osc_server.BlockingOSCUDPServer((server_ip, act_confirm_server_port),
                                                              act_confirm_dispatcher)
        # Configuration
        config_dispatcher = Dispatcher()
        config_dispatcher.map("/config", self.config_handler)

        # config_server = osc_server.ThreadingOSCUDPServer((server_ip, config_server_port), config_dispatcher)
        config_server = osc_server.BlockingOSCUDPServer((server_ip, config_server_port), config_dispatcher)

        # Excitor raw action
        excitor_raw_act_dispatcher = Dispatcher()
        excitor_raw_act_dispatcher.map("/Excitor_Act/Uni_Sim/*", self.excitor_raw_act_from_uni_sim_handler)
        # excitor_raw_act_server = osc_server.ThreadingOSCUDPServer((server_ip, excitor_raw_act_server_port),
        #                                                           excitor_raw_act_dispatcher)
        excitor_raw_act_server = osc_server.BlockingOSCUDPServer((server_ip, excitor_raw_act_server_port),
                                                                 excitor_raw_act_dispatcher)

        return obs_server, act_confirm_server, config_server, excitor_raw_act_server

    def serve_obs_server(self, serve_time=1):
        serve_thread = Thread(target=self.obs_server.serve_forever)
        serve_thread.start()
        time.sleep(serve_time)
        self.obs_server.shutdown()

    def serve_excitor_raw_act_server(self, serve_time=1):
        serve_thread = Thread(target=self.excitor_raw_act_server.serve_forever)
        serve_thread.start()
        time.sleep(serve_time)
        self.excitor_raw_act_server.shutdown()

    def serve_act_confirm_server(self, serve_time=1):
        serve_thread = Thread(target=self.act_confirm_server.serve_forever)
        serve_thread.start()
        time.sleep(serve_time)
        self.act_confirm_server.shutdown()

    def serve_config_server(self, serve_time=1):
        serve_thread = Thread(target=self.config_server.serve_forever)
        serve_thread.start()
        time.sleep(serve_time)
        self.config_server.shutdown()

    def start_obs_server(self):
        """
        Start receiving message.
        """
        # print("Internal_env.comm_manager.obs_server serving on {}".format(self.obs_server.server_address))
        serve_thread = Thread(target=self.obs_server.serve_forever)
        serve_thread.start()
        return serve_thread.is_alive()

    def pause_obs_server(self):
        """
        Stop receiving message.
        """
        # print("Pause Internal_env.comm_manager.obs_server serving on {}".format(self.obs_server.server_address))
        self.obs_server.shutdown()

    def close_obs_server(self):
        """
        Clean up server.
        """
        self.obs_server.server_close()

    def start_act_confirm_server(self):
        """
        Start receiving message.
        """
        # print("Internal_env.comm_manager.act_confirm_server serving on {}".format(self.act_confirm_server.server_address))
        serve_thread = Thread(target=self.act_confirm_server.serve_forever)
        serve_thread.start()

    def pause_act_confirm_server(self):
        """
        Stop receiving message.
        """
        # print("Pause Internal_env.comm_manager.act_confirm_server serving on {}".format(self.act_confirm_server.server_address))
        pause_thread = Thread(target=self.act_confirm_server.shutdown)
        pause_thread.start()

    def close_act_confirm_server(self):
        """
        Clean up server.
        """
        self.act_confirm_server.server_close()

    def start_config_server(self):
        """
        Start receiving message.
        """
        print("Internal_env.comm_manager.config_server: serving on {}".format(self.config_server.server_address))
        serve_thread = Thread(target=self.config_server.serve_forever)
        serve_thread.start()

    def pause_config_server(self):
        """
        Stop receiving message.
        """
        print("Pause Internal_env.comm_manager.config_server serving on {}".format(self.config_server.server_address))
        pause_thread = Thread(target=self.config_server.shutdown)
        pause_thread.start()

    def close_config_server(self):
        """
        Clean up server.
        """
        self.config_server.server_close()

    def start_excitor_raw_act_server(self):
        serve_thread = Thread(target=self.excitor_raw_act_server.serve_forever)
        serve_thread.start()
        return serve_thread.is_alive()

    def pause_excitor_raw_act_server(self):
        """
        Stop receiving message.
        """
        # print("Pause Internal_env.comm_manager.obs_server serving on {}".format(self.obs_server.server_address))
        self.excitor_raw_act_server.shutdown()

    def close_excitor_raw_act_server(self):
        """
        Clean up server.
        """
        self.excitor_raw_act_server.server_close()

    def _create_client(self):
        """
        Create client for Unity Simulator and Processing Simulator.
        :return:
        """
        unity_client_ip = self.config['LAS_Unity_Sim']['IP']
        unity_client_port = self.config['LAS_Unity_Sim']['Port']
        unity_client = udp_client.SimpleUDPClient(unity_client_ip, unity_client_port)

        processing_client_ip = self.config['LAS_Processing_Sim']['IP']
        processing_client_port = self.config['LAS_Processing_Sim']['Port']
        processing_client = udp_client.SimpleUDPClient(processing_client_ip, processing_client_port)

        gui_osc_server_client_ip = self.config['LAS_GUI_OSC_Server']['IP']
        gui_osc_server_client_port = self.config['LAS_GUI_OSC_Server']['Port']
        gui_osc_server_client = udp_client.SimpleUDPClient(gui_osc_server_client_ip, gui_osc_server_client_port)

        return unity_client, processing_client, gui_osc_server_client

    def _create_video_capture_client(self):
        video_capture_client_ip = self.config['LAS_Unity_Sim']['IP']
        video_capture_client_port = self.config['LAS_Unity_Sim']['Video_Capture_Port']
        video_capture_client = udp_client.SimpleUDPClient(video_capture_client_ip, video_capture_client_port)
        return video_capture_client

    def _create_excitor_client(self):
        """
        Create client for controlling excitor.
        :return:
        """
        excitor_client_ip = self.config['LAS_Unity_Sim']['IP']
        excitor_client_port = self.config['LAS_Unity_Sim']['Excitor_Port']
        excitor_client = udp_client.SimpleUDPClient(excitor_client_ip, excitor_client_port)
        return excitor_client

    def _create_node_client_for_raw_act_execution(self):
        """ Each node_client sends osc message to a specific port."""
        node_client = {}
        for source_server in self.config['LAS_ML_Agent']['potential_raw_actuator_control_destination']:
            # Actuator Obs and Confirmation
            node_client[source_server] = {}
            for device_type in self.actuator_dict:
                for device_id in self.actuator_dict[device_type]:
                    device_node_id, device, device_num, device_pin = device_id.split('_')
                    if device_node_id not in node_client[source_server]:
                        client_port = int('5' + device_node_id[-4:])
                        if source_server == 'Uni_Sim':
                            client_ip = self.config['LAS_Unity_Sim']['IP']
                        elif source_server == 'Pro_Sim':
                            client_ip = self.config['LAS_Processing_Sim']['IP']
                        else:
                            raise ValueError("client_ip is wrong!")
                        node_client[source_server][device_node_id] = udp_client.SimpleUDPClient(client_ip,
                                                                                                client_port)
        return node_client


class LASIntlEnv(object):
    """Living Architecture System Internal Environment"""
    def __init__(self, config, rew_comp=None):
        # Initialize configuration info
        self.config = config
        self.las_config = config.las_config
        self.internal_env_config = config.internal_env_config
        self.comm_manager_config = config.comm_manager_config

        # Initialize device locator
        self.device_locator = pd.read_csv(self.las_config['device_locator_csv'])

        # Init comm_manager to None, and instantiate it only in reset().
        self.comm_manager = None

        # Init action space
        self.act_val_max = self.internal_env_config['action_space']['act_value']['act_val_max']
        self.act_val_min = self.internal_env_config['action_space']['act_value']['act_val_min']
        self.employed_act_space = self.internal_env_config['action_space']['employed_action_space']
        self.wait_time_for_action_execution = self.internal_env_config['wait_time_for_action_execution']

        #   Init raw action space
        self.raw_act_space_dict, self.raw_act_space_list, self.raw_act_space_dim = self._init_raw_action_space()

        #   Init parameterized action space
        self.para_act_space_dict, self.para_act_space_list, self.para_act_space_dim = self._init_para_action_space()

        if self.employed_act_space == 'raw_action_space':
            self.act_dim = self.raw_act_space_dim
        elif self.employed_act_space == 'para_action_space':
            self.act_dim = self.para_act_space_dim
        else:
            raise ValueError("Wrong employed_act_space value: {}".format(self.employed_act_space))
        # Crucial: action space value range [-1,1]
        self.action_space = gym.spaces.Box(np.array([self.act_val_min for i in range(self.act_dim)], dtype="float32"),
                                           np.array([self.act_val_max for i in range(self.act_dim)], dtype="float32"))

        # Init obsservation space
        self.add_velocity_to_obs = self.internal_env_config['obs_space']['add_velocity_to_obs']
        self.add_act_to_obs = self.internal_env_config['obs_space']['add_act_to_obs']
        self.add_past_obs_to_obs = self.internal_env_config['obs_space']['add_past_obs_to_obs']

        self.proprio_and_extero_obs_space_dict, self.proprio_and_extero_obs_space_list, self.proprio_and_extero_obs_space_dim, \
        self.extero_obs_space_dict, self.extero_obs_space_list, self.extero_obs_space_dim, \
        self.proprio_obs_space_dict, self.proprio_obs_space_list, self.proprio_obs_space_dim = self._init_observation_space()
        # Save index of proprioceptor in NorthRiver and SouthRiver separately for a new reward function
        self.nr_proprio_idx = []
        self.sr_proprio_idx = []
        self.mg_proprio_idx = []
        self.tg_proprio_idx = []
        for obs_idx, obs_item in enumerate(self.proprio_obs_space_list):
            if 'NR' in obs_item:
                self.nr_proprio_idx.append(obs_idx)
            if 'SR' in obs_item:
                self.sr_proprio_idx.append(obs_idx)
            if 'MG' in obs_item:
                self.mg_proprio_idx.append(obs_idx)
            if 'TG' in obs_item:
                self.tg_proprio_idx.append(obs_idx)

        self.obs_dim = self.proprio_and_extero_obs_space_dim
        # Determine if add velocity to observation
        if self.add_velocity_to_obs:
            self.obs_dim = self.obs_dim + self.proprio_and_extero_obs_space_dim
        # Determine if add action to observation
        if self.add_act_to_obs:
            self.obs_dim = self.obs_dim + self.act_dim
        # Determine if add past observation to new observation (Note: setup this after add_velocity_to_obs and add_act_to_obs)
        if self.add_past_obs_to_obs:
            self.obs_dim = self.obs_dim*2
        self.observation_space = gym.spaces.Box(np.array([0 for i in range(self.obs_dim)], dtype="float32"),
                                                np.array([1 for i in range(self.obs_dim)], dtype="float32"))

        # Init reward function
        self.reward_type = self.internal_env_config['reward_function']['reward_type']
        self.handcrafted_reward_type = self.internal_env_config['reward_function']['handcrafted_reward_type']
        self.hc_reward_range = self.internal_env_config['reward_function']['hc_reward_range']
        self.rew_comp = None  # rew_comp  # TODO: load reward component

        # Variables used to save past obs and info
        self.obs, self.info = None, None
        self.obs_traj, self.act_traj, self.obs2_traj = [], [], []

        # Setup maximum episode steps
        self._max_episode_steps = self.internal_env_config["max_episode_steps"]
        self.fps = 1/(self.internal_env_config["time_window_for_obs_collection"]+self.internal_env_config['wait_time_for_action_execution'])

    def set_reward_component(self, rew_comp):
        self.rew_comp = rew_comp

    def reset(self):
        # Only init communication in reset to avoid port occupation problem.
        if self.comm_manager is None:
            self.comm_manager = CommunicationManager(self.comm_manager_config)
        # Turn off exciter
        self.comm_manager.excitor_client.send_message("/Excitor/Enable", "false")
        # TODO: Start video capture and set behavior mode to RL policy

        # Collect current proprioception and exteroception observation
        new_proprio_and_extero_obs, info = self._collect_and_construct_proprio_and_extero_obs()
        new_obs_ts = datetime.now()  # new_obs timestamp
        missing_data_count = 0
        while info["obs_missing_data"]:
            new_proprio_and_extero_obs, info = self._collect_and_construct_proprio_and_extero_obs()
            missing_data_count += 1
            print('Warning! missing_data_count={}'.format(missing_data_count))

        new_obs = new_proprio_and_extero_obs
        # Determine if add velocity to observation
        if self.add_velocity_to_obs:
            new_obs = np.concatenate((new_obs, np.zeros(len(new_proprio_and_extero_obs))))  # Add zero for the initial observation
        # Determine if add action to observation
        if self.add_act_to_obs:
            new_obs = np.concatenate((new_obs, np.zeros(self.act_dim)))  # Add zero for the initial observation
        # Determine if add past observation to new observation (Note: setup this after add_velocity_to_obs and add_act_to_obs)
        if self.add_past_obs_to_obs:
            new_obs = np.concatenate((new_obs, np.zeros(len(new_obs))))

        self.obs, self.proprio_and_extero_obs, self.info = new_obs, new_proprio_and_extero_obs, info
        self.info = {'obs_datetime': new_obs_ts}
        return new_obs, self.info

    def _init_raw_action_space(self):
        """Initialize raw action space"""
        raw_act_space_dict = {}
        raw_act_space_list = []
        checked_device = []
        for index, row in self.device_locator.iterrows():
            # Get device info
            device_group = row['GROUP'].split(':')[0]
            device_type = row['DEVICE_TYPE']
            device = row['DEVICE']
            device_node_id = str(row['NODE ID'])
            device_num = row['NUM']
            device_uid = str(row['UID'])
            device_config = row['CONFIG']
            if type(device_config) == str:
                device_bottom_pin = device_config.split(' ')[1].split(';')[0]
            else:
                device_bottom_pin = "-1"

            # Avoid initialize observation for duplicated device
            device_name = "{}_{}_{}_{}".format(device_group, device_node_id, device, device_num, device_uid)
            if device_name not in checked_device:
                checked_device.append(device_name)

                # Only consider device with proper device_group
                if device_group in self.internal_env_config['action_space']['raw_act_space']['device_group']:
                    if device_group not in raw_act_space_dict:
                        raw_act_space_dict[device_group] = {}
                    if device_node_id not in raw_act_space_dict[device_group]:
                        raw_act_space_dict[device_group][device_node_id] = {}
                    # Only consider actuators listed in raw_act_space_config['actuator_type']
                    if device in self.internal_env_config['action_space']['raw_act_space']['actuator_type']:
                        # Add PIN as device_id
                        device_id = "{}_{}_{}_{}_{}".format(device_group, device_node_id, device, device_num,
                                                            device_uid)
                        if device_id not in raw_act_space_dict[device_group][device_node_id]:
                            raw_act_space_dict[device_group][device_node_id][device_id] = "Top PIN"
                            raw_act_space_list.append(device_id)
                        # Add BottonPin as device_id (actuator in bottomPin is treated as an independent actuator)
                        if type(device_config) == str:
                            device_bottom_pin = device_config.split(' ')[1].split(';')[0]
                            device_id = "{}_{}_{}_{}".format(device_group, device_node_id, device, device_num,
                                                             device_bottom_pin)
                            if device_id not in raw_act_space_dict[device_group][device_node_id]:
                                raw_act_space_dict[device_group][device_node_id][device_id] = "Bottom PIN"
                                raw_act_space_list.append(device_id)
        return raw_act_space_dict, raw_act_space_list, len(raw_act_space_list)

    def _init_para_action_space(self):
        """TODO: Initialize parameterized action space."""
        para_act_space_dict = {}
        para_act_space_list = []
        for behavior_name, behavior_val in self.internal_env_config['action_space']['para_act_space'].items():
            for param_name, param_val in behavior_val.items():
                value_prefix = '{} {}'.format(behavior_name, param_name)
                para_act_space_dict[value_prefix] = {}
                para_act_space_list.append(value_prefix)
        para_act_space_dim = len(para_act_space_list)
        return para_act_space_dict, para_act_space_list, para_act_space_dim

    def _init_observation_space(self):
        """Initialize observation space"""
        extero_obs_space_dict = {}
        extero_obs_space_list = []
        proprio_obs_space_dict = {}
        proprio_obs_space_list = []

        checked_device = []
        for index, row in self.device_locator.iterrows():
            # Get device info
            device_group = row['GROUP'].split(':')[0]
            device_type = row['DEVICE_TYPE']
            device = row['DEVICE']
            device_node_id = str(row['NODE ID'])
            device_num = row['NUM']
            device_uid = str(row['UID'])
            # If device_uid == '--', it is neither a sensor nor an actuator
            if device_uid == '--':
                continue
            device_config = row['CONFIG']
            if type(device_config) == str:
                device_bottom_pin = device_config.split(' ')[1].split(';')[0]
            else:
                device_bottom_pin = "-1"

            # Avoid initialize observation for duplicated device
            device_name = "{}_{}_{}_{}".format(device_group, device_node_id, device, device_num, device_uid)
            if device_name not in checked_device:
                checked_device.append(device_name)
                # Observation from exteroception i.e. sensor data
                #  only consider device in proper group
                if device_group in self.internal_env_config['obs_space']['exteroception']['device_group']:
                    if device_group not in extero_obs_space_dict:
                        extero_obs_space_dict[device_group] = {}
                    if device_node_id not in extero_obs_space_dict[device_group]:
                        extero_obs_space_dict[device_group][device_node_id] = {}
                    # only consider device with proper device type
                    if device in self.internal_env_config['obs_space']['exteroception']['sensor_type']:
                        data_size = self.internal_env_config['obs_space']['exteroception']['sensor_type'][device][
                            "size"]
                        if self.internal_env_config['obs_space']['exteroception'][
                            'obs_construction_method'] == "concatenate":
                            # Concatenate observations downsampled to fixed frequency in a given window
                            sample_size = math.ceil(
                                self.internal_env_config['obs_space']['exteroception']['obs_frequency'] *
                                self.internal_env_config['time_window_for_obs_collection'])
                            # sample_size = self.internal_env_config['obs_space']['exteroception']['obs_frequency']
                            for obs_i in range(sample_size):
                                # treat each element of the sensory data of a sensor as an entry in observation space
                                for element_i in range(data_size):
                                    obs_id = "{}_{}_{}_{}_{}_obs{}_ele{}".format(device_group, device_node_id, device,
                                                                                 device_num, device_uid,
                                                                                 obs_i, element_i)
                                    extero_obs_space_dict[device_group][device_node_id][obs_id] = obs_id
                                    extero_obs_space_list.append(obs_id)
                        elif self.internal_env_config['obs_space']['exteroception'][
                            'obs_construction_method'] == "average":
                            # Average observations downsampled to fixed frequency in a given window
                            #  treat each element of the sensory data of a sensor as an entry in observation space
                            for element_i in range(data_size):
                                obs_id = "{}_{}_{}_{}_{}_avg_ele{}".format(device_group, device_node_id, device,
                                                                           device_num, device_uid,
                                                                           element_i)
                                extero_obs_space_dict[device_group][device_node_id][obs_id] = obs_id
                                extero_obs_space_list.append(obs_id)
                        else:
                            raise ValueError("Please choose proper obs_construction_method!")

                # Observation from proprioception i.e. actuator state
                #  only consider device in proper group
                if device_group in self.internal_env_config['obs_space']['proprioception']['device_group']:
                    if device_group not in proprio_obs_space_dict:
                        proprio_obs_space_dict[device_group] = {}
                    if device_node_id not in proprio_obs_space_dict[device_group]:
                        proprio_obs_space_dict[device_group][device_node_id] = {}
                    # only consider device with proper device type
                    if device in self.internal_env_config['obs_space']['proprioception']['actuator_type']:
                        data_size = self.internal_env_config['obs_space']['proprioception']['actuator_type'][device][
                            "size"]
                        # Add PIN as device_id
                        if self.internal_env_config['obs_space']['proprioception'][
                            'obs_construction_method'] == "concatenate":
                            # Concatenate observations downsampled to fixed frequency in a given window
                            sample_size = math.ceil(
                                self.internal_env_config['obs_space']['proprioception']['obs_frequency'] *
                                self.internal_env_config['time_window_for_obs_collection'])
                            # sample_size = self.internal_env_config['obs_space']['proprioception']['obs_frequency']
                            for obs_i in range(sample_size):
                                # treat each element of the actuator state of a actuator as an entry in observation space
                                for element_i in range(data_size):
                                    obs_id = "{}_{}_{}_{}_{}_obs{}_ele{}".format(device_group, device_node_id, device,
                                                                                 device_num,
                                                                                 device_uid, obs_i, element_i)
                                    proprio_obs_space_dict[device_group][device_node_id][obs_id] = obs_id
                                    proprio_obs_space_list.append(obs_id)
                        elif self.internal_env_config['obs_space']['proprioception'][
                            'obs_construction_method'] == "average":
                            # Average observations downsampled to fixed frequency in a given window
                            #  treat each element of the sensory data of a sensor as an entry in observation space
                            for element_i in range(data_size):
                                obs_id = "{}_{}_{}_{}_{}_avg_ele{}".format(device_group, device_node_id, device,
                                                                           device_num, device_uid,
                                                                           element_i)
                                proprio_obs_space_dict[device_group][device_node_id][obs_id] = obs_id
                                proprio_obs_space_list.append(obs_id)
                        else:
                            raise ValueError("Please choose proper obs_construction_method!")
                        # Add BottonPin as device_id (actuator in bottomPin is treated as an independent actuator)
                        if type(device_config) == str:
                            device_bottom_pin = device_config.split(' ')[1].split(';')[0]
                            if self.internal_env_config['obs_space']['proprioception'][
                                'obs_construction_method'] == "concatenate":
                                # Concatenate observations downsampled to fixed frequency in a given window
                                sample_size = math.ceil(
                                    self.internal_env_config['obs_space']['proprioception']['obs_frequency'] *
                                    self.internal_env_config['time_window_for_obs_collection'])
                                # sample_size = self.internal_env_config['obs_space']['proprioception']['obs_frequency']
                                for obs_i in range(sample_size):
                                    # treat each element of the actuator state of a actuator as an entry in observation space
                                    for element_i in range(data_size):
                                        obs_id = "{}_{}_{}_{}_{}_obs{}_ele{}".format(device_group, device_node_id,
                                                                                     device, device_num,
                                                                                     device_bottom_pin, obs_i,
                                                                                     element_i)
                                        proprio_obs_space_dict[device_group][device_node_id][obs_id] = obs_id
                                        proprio_obs_space_list.append(obs_id)
                            elif self.internal_env_config['obs_space']['proprioception'][
                                'obs_construction_method'] == "average":
                                # Average observations downsampled to fixed frequency in a given window
                                #  treat each element of the sensory data of a sensor as an entry in observation space
                                for element_i in range(data_size):
                                    obs_id = "{}_{}_{}_{}_{}_avg_ele{}".format(device_group, device_node_id, device,
                                                                               device_num, device_bottom_pin, element_i)
                                    proprio_obs_space_dict[device_group][device_node_id][obs_id] = obs_id
                                    proprio_obs_space_list.append(obs_id)
                            else:
                                raise ValueError("Please choose proper obs_construction_method!")

        proprio_and_extero_obs_space_dict = {**extero_obs_space_dict, **proprio_obs_space_dict}
        proprio_and_extero_obs_space_list = extero_obs_space_list + proprio_obs_space_list
        extero_obs_space_dim = len(extero_obs_space_list)
        proprio_obs_space_dim = len(proprio_obs_space_list)
        proprio_and_extero_obs_space_dim = len(proprio_and_extero_obs_space_list)
        return proprio_and_extero_obs_space_dict, proprio_and_extero_obs_space_list, proprio_and_extero_obs_space_dim, \
               extero_obs_space_dict, extero_obs_space_list, extero_obs_space_dim, \
               proprio_obs_space_dict, proprio_obs_space_list, proprio_obs_space_dim

    def _execute_raw_action(self, action):
        """Execute raw action."""
        # Check action length
        if len(action) != self.raw_act_space_dim:
            raise ValueError("Length of raw action does not match raw_act_space_dim!")
        # Normalize action to [0,1] for actuators
        action = (action - self.act_val_min) / (self.act_val_max - self.act_val_min)
        # Send
        raw_act_execute_server = self.internal_env_config['action_space']['raw_act_space']['LAS_for_action_execution']
        act_send_index = 0
        for device_group in self.raw_act_space_dict:
            for device_node_id in self.raw_act_space_dict[device_group]:
                address = '/MLAgent/RAW_OUTPUT/{}'.format(device_node_id)
                # Only consider node with actuators and construct message in format: "pin float pin float ..."
                if len(self.raw_act_space_dict[device_group][device_node_id]) != 0:
                    value = []
                    for device_name in self.raw_act_space_dict[device_group][device_node_id]:
                        _, _, device, device_num, device_pin = device_name.split('_')
                        value += [int(device_pin), float(action[act_send_index])]
                        act_send_index += 1
                    self.comm_manager.node_client[raw_act_execute_server][device_node_id].send_message(address, value)

    def _execute_para_action(self, action):
        """Execute parameterized action."""
        address = '/setDatParameter'
        # Check action length
        if len(action) != self.para_act_space_dim:
            raise ValueError("Length of parameterized action does not match para_act_space_dim!")
        # Convert and send parameterized action
        for act_i, act_name in enumerate(self.para_act_space_list):
            behavior_name, param_name = act_name.split(' ')
            param_type = self.internal_env_config['action_space']['para_act_space'][behavior_name][param_name]['type']
            # Convert parameterized action to valid values
            param_min = self.internal_env_config['action_space']['para_act_space'][behavior_name][param_name]['min']
            param_max = self.internal_env_config['action_space']['para_act_space'][behavior_name][param_name]['max']
            if param_type == 'bool':
                if action[act_i] >= 0:
                    act_value = True
                else:
                    act_value = False
            elif param_type == 'float':
                act_value = (action[act_i] - self.act_val_min) * (param_max - param_min) / (self.act_val_max - self.act_val_min) + param_min
            elif param_type == 'int':
                act_value = int((action[act_i] - self.act_val_min) * (param_max - param_min) / (self.act_val_max - self.act_val_min) + param_min)
            else:
                raise ValueError('Unrecognized parameter type!')
            # Send message
            if behavior_name == "AmbientWaves":
                self.comm_manager.gui_osc_server_client.send_message(address, "{} {} {}".format("AmbientWaves/Wave_SR_1", param_name, act_value))
                self.comm_manager.gui_osc_server_client.send_message(address, "{} {} {}".format("AmbientWaves/Wave_NR_3", param_name, act_value))
            else:
                self.comm_manager.gui_osc_server_client.send_message(address, "{} {} {}".format(behavior_name, param_name, act_value))

    def _collect_raw_action_execution_confirmation(self):
        # Collect confirmation
        act_execution_server = self.internal_env_config['action_space']['raw_act_space']['LAS_for_action_execution']
        raw_act_received_flag = True
        raw_act_confirm = []
        for raw_act_label in self.raw_act_space_list:
            device_group, device_node_id, device, device_num, device_pin = raw_act_label.split('_')
            device_name = "{}_{}_{}_{}".format(device_node_id, device, device_num, device_pin)
            act_confirm_buff = self.comm_manager.actuator_confirm_buffer[act_execution_server][device][device_name]
            if act_confirm_buff.empty():
                raw_act_confirm.append("not received")
                raw_act_received_flag = False
            else:
                raw_act_confirm.append(act_confirm_buff.get())
        return raw_act_received_flag, raw_act_confirm

    def _collect_and_construct_proprio_and_extero_obs(self):
        """Construct observation."""
        extero_obs_source_server = self.internal_env_config['obs_space']['exteroception']['obs_source_server']
        proprio_obs_source_server = self.internal_env_config['obs_space']['proprioception']['obs_source_server']

        ########################################################################################
        #                                Collect message                                       #
        #   1. Empty observation replay buffer                                                 #
        #   2. Resume server                                                                   #
        #   3. Wait a specific time window for collecting message                              #
        #   4. Pause server                                                                    #
        ########################################################################################
        self.comm_manager.empty_obs_buffer()
        # Turn on obs_server for a specific time window
        self.comm_manager.serve_obs_server(serve_time=self.internal_env_config['time_window_for_obs_collection'])

        extracted_device = []  # Record extracted device to avoid duplicated extraction
        obs_missing_data = False
        # Extract and construct observation for exteroception
        extero_obs_list = []
        for extero_obs_i in self.extero_obs_space_list:
            device_group, device_node_id, device, device_num, device_uid, obs_i, ele_i = extero_obs_i.split('_')
            device_id = "{}_{}_{}_{}".format(device_node_id, device, device_num, device_uid)
            # Each device is only extracted once for observation
            if device_id not in extracted_device:
                extracted_device.append(device_id)
                size = self.comm_manager.sensor_obs_buffer[extero_obs_source_server][device][device_id].qsize()
                # print("{}: {}".format(device_id, size))
                tmp_integrated_obs = []
                # Check buffer size
                sample_size = math.ceil(
                    self.internal_env_config['obs_space']['exteroception']['obs_frequency'] * self.internal_env_config[
                        'time_window_for_obs_collection'])
                if size < sample_size:
                    obs_missing_data = True
                    data_size = self.internal_env_config['obs_space']['exteroception']['sensor_type'][device]["size"]
                    if self.internal_env_config['obs_space']['exteroception']['obs_construction_method'] == "average":
                        for element_i in range(data_size):
                            tmp_integrated_obs.append('missing data: buff_size={}'.format(size))
                    elif self.internal_env_config['obs_space']['exteroception'][
                        'obs_construction_method'] == "concatenate":
                        for obs_i in range(sample_size):
                            for element_i in range(data_size):
                                tmp_integrated_obs.append('missing data: buff_size={}'.format(size))
                    else:
                        raise ValueError("Please check obs_construction_type.")
                else:
                    # Because message sending is not exactly at a fixed frequency, we need to down sample received
                    #   message to a fixed length by evenly splitting the range to sub-windows, and pick the last one in each sub-window.
                    downsample_index = []
                    for sub_window in np.array_split(np.arange(size), sample_size):
                        downsample_index.append(sub_window[-1])
                    # Integrate samples in the time window
                    for i in range(size):
                        data_point = self.comm_manager.sensor_obs_buffer[extero_obs_source_server][device][
                            device_id].get()
                        if type(data_point) != list:
                            data_point = [data_point]

                        if i in downsample_index:
                            if i == downsample_index[0]:
                                if self.internal_env_config['obs_space']['exteroception'][
                                    'obs_construction_method'] == "average":
                                    tmp_integrated_obs = np.divide(data_point, sample_size)
                                elif self.internal_env_config['obs_space']['exteroception'][
                                    'obs_construction_method'] == "concatenate":
                                    tmp_integrated_obs = data_point
                                else:
                                    raise ValueError("Please check obs_construction_type.")
                            else:
                                if self.internal_env_config['obs_space']['exteroception'][
                                    'obs_construction_method'] == "average":
                                    avg_data_point = np.divide(data_point, sample_size)
                                    tmp_integrated_obs = np.add(tmp_integrated_obs, avg_data_point).tolist()
                                elif self.internal_env_config['obs_space']['exteroception'][
                                    'obs_construction_method'] == "concatenate":
                                    tmp_integrated_obs = np.concatenate((tmp_integrated_obs, data_point)).tolist()
                                else:
                                    raise ValueError("Please check obs_construction_type.")
                extero_obs_list.extend(tmp_integrated_obs)
        # Extract and construct observation for proprioception
        proprio_obs_list = []
        for proprio_obs_i in self.proprio_obs_space_list:
            device_group, device_node_id, device, device_num, device_uid, obs_i, ele_i = proprio_obs_i.split('_')
            device_id = "{}_{}_{}_{}".format(device_node_id, device, device_num, device_uid)

            # Each device is only extracted once for observation
            if device_id not in extracted_device:
                extracted_device.append(device_id)
                size = self.comm_manager.actuator_obs_buffer[proprio_obs_source_server][device][device_id].qsize()
                # print("{}: {}".format(device_id, size))
                tmp_integrated_obs = []
                # Check buffer size
                sample_size = math.ceil(
                    self.internal_env_config['obs_space']['proprioception']['obs_frequency'] * self.internal_env_config[
                        'time_window_for_obs_collection'])
                if size < sample_size:
                    obs_missing_data = True
                    data_size = self.internal_env_config['obs_space']['proprioception']['actuator_type'][device]["size"]
                    if self.internal_env_config['obs_space']['proprioception']['obs_construction_method'] == "average":
                        for element_i in range(data_size):
                            tmp_integrated_obs.append('missing data: buff_size={}'.format(size))
                    elif self.internal_env_config['obs_space']['proprioception'][
                        'obs_construction_method'] == "concatenate":
                        for obs_i in range(sample_size):
                            for element_i in range(data_size):
                                tmp_integrated_obs.append('missing data: buff_size={}'.format(size))
                    else:
                        raise ValueError("Please check obs_construction_type.")
                else:
                    # Because message sending is not exactly at a fixed frequency, we need to down sample received
                    #   message to a fixed length.
                    downsample_index = []
                    for sub_window in np.array_split(np.arange(size), sample_size):
                        downsample_index.append(sub_window[-1])
                    # Integrate samples in the time window
                    for i in range(size):
                        data_point = self.comm_manager.actuator_obs_buffer[proprio_obs_source_server][device][
                            device_id].get()
                        if type(data_point) != list:
                            data_point = [data_point]

                        if i in downsample_index:
                            if i == downsample_index[0]:
                                if self.internal_env_config['obs_space']['proprioception'][
                                    'obs_construction_method'] == "average":
                                    tmp_integrated_obs = np.divide(data_point, sample_size)
                                elif self.internal_env_config['obs_space']['proprioception'][
                                    'obs_construction_method'] == "concatenate":
                                    tmp_integrated_obs = data_point
                                else:
                                    raise ValueError("Please check obs_construction_type.")
                            else:
                                if self.internal_env_config['obs_space']['proprioception'][
                                    'obs_construction_method'] == "average":
                                    avg_data_point = np.divide(data_point, sample_size)
                                    tmp_integrated_obs = np.add(tmp_integrated_obs, avg_data_point).tolist()
                                elif self.internal_env_config['obs_space']['proprioception'][
                                    'obs_construction_method'] == "concatenate":
                                    tmp_integrated_obs = np.concatenate((tmp_integrated_obs, data_point)).tolist()
                                else:
                                    raise ValueError("Please check obs_construction_type.")
                proprio_obs_list.extend(tmp_integrated_obs)
        # Combine observation for exterp- and proprioception
        proprio_and_extero_obs = np.asarray(proprio_obs_list + extero_obs_list)
        info = {"obs_missing_data": obs_missing_data,
                "extero_obs_list": extero_obs_list, "proprio_obs_list": proprio_obs_list}

        return proprio_and_extero_obs, info

    def _calculate_handcrafted_reward(self, obs, act, new_obs, info):
        """Define handcrafted reward function"""
        extero_obs = np.asarray(info['extero_obs_list'])
        proprio_obs = np.asarray(info['proprio_obs_list'])
        # Calculate reward
        if self.handcrafted_reward_type == 'active_all':
            # Reward active behavior in the whole sculpture
            reward = np.mean(proprio_obs)
        elif self.handcrafted_reward_type == 'calm_all':
            # Reward calm behavior in the whole sculpture
            reward = -np.mean(proprio_obs)
        elif self.handcrafted_reward_type == 'active_NR_calm_SR':
            # Reward active behavior in NorthRiver and calm behavior in SouthRiver
            reward = np.mean(proprio_obs[self.nr_proprio_idx]) - np.mean(proprio_obs[self.sr_proprio_idx])
        elif self.handcrafted_reward_type == 'active_SR_calm_NR':
            # Reward active behavior in SouthRiver and calm behavior in NorthRiver
            reward = -np.mean(proprio_obs[self.nr_proprio_idx]) + np.mean(proprio_obs[self.sr_proprio_idx])
        else:
            raise ValueError("handcrafted_reward_type: {} is not defined!".format(self.handcrafted_reward_type))

        # Shift the reward to different value ranges
        reward_range_0_pos_1 = reward
        reward_range_neg_1_pos_1 = reward*2-1
        reward_range_0_pos_2 = reward*2
        reward_range_neg_2_pos_2 = reward * 4 - 2
        reward_range_0_pos_10 = reward * 10
        reward_range_0_pos_100 = reward * 100
        return reward_range_0_pos_1, reward_range_neg_1_pos_1, reward_range_0_pos_2, reward_range_neg_2_pos_2, reward_range_0_pos_10, reward_range_0_pos_100

    def step(self, action):
        """
        Function used to interact with learning agent where each step takes action given by agent, collects
        observations within a time window, calculate reward, determine if the task is done, and return
        diagnosing information.
        Note: if info.obs_missing_data == true, the experience (obs, act, rew, new_obs, done, info) should be thrown.
        :param action: action decision for observation
        :return:
        """
        # 1. Check validity of action and clip action to proper range
        if (self.employed_act_space == 'raw_action_space' and len(action) != self.raw_act_space_dim) or (
                self.employed_act_space == 'para_action_space' and len(action) != self.para_act_space_dim):
            raise ValueError(
                "Length of action {} does not match action dimension {}!".format(len(action), self.raw_act_space_dim))

        action = np.clip(action, self.act_val_min, self.act_val_max)

        # 2. Resume server and empty buffer, which is prepared for collecting action confirmation.
        self.comm_manager.empty_all_buffers()
        self.comm_manager.start_act_confirm_server()

        # 3. Execute action
        if self.employed_act_space == 'raw_action_space':
            self._execute_raw_action(action)
        elif self.employed_act_space == 'para_action_space':
            self._execute_para_action(action)
        else:
            raise ValueError("Wrong employed_act_space!")
        act_ts = datetime.now()     # Action execution timestamp

        # 4. Wait for executing action and start receiving sensors and actuators state
        time.sleep(self.wait_time_for_action_execution)

        # 5. Collect and construct observation
        # Collect current proprioception and exteroception observation
        new_proprio_and_extero_obs, info = self._collect_and_construct_proprio_and_extero_obs()
        new_obs_ts = datetime.now()  # new_obs timestamp
        missing_data_count = 0
        while info["obs_missing_data"]:
            new_proprio_and_extero_obs, info = self._collect_and_construct_proprio_and_extero_obs()
            missing_data_count += 1
            print('Warning! missing_data_count={}'.format(missing_data_count))

        new_obs = new_proprio_and_extero_obs
        # Determine if add velocity to observation
        if self.add_velocity_to_obs:
            new_obs = np.concatenate((new_obs, new_proprio_and_extero_obs-self.proprio_and_extero_obs))  # Add zero for the initial observation
        # Determine if add action to observation
        if self.add_act_to_obs:
            new_obs = np.concatenate((new_obs, action))  # Add zero for the initial observation
        # Determine if add past observation to new observation (Note: setup this after add_velocity_to_obs and add_act_to_obs)
        if self.add_past_obs_to_obs:
            new_obs = np.concatenate((new_obs, self.obs[-int(len(self.obs)/2):]))  # Note: only add the second half of the past observation

        # Add local trajectory
        self.obs_traj.append(self.obs.reshape(1, -1))
        self.act_traj.append(action.reshape(1, -1))
        self.obs2_traj.append(new_obs.reshape(1, -1))

        # 6. Collect action execution confirmation
        #   Note: Collect action execution confirmations after observation to earn enough time for receiving them.
        self.comm_manager.pause_act_confirm_server()
        raw_act_received_flag, raw_act_confirm = self._collect_raw_action_execution_confirmation()
        info["raw_act_received_flag"] = raw_act_received_flag
        info["raw_act_confirm"] = raw_act_confirm

        rew, rew_ts, done, done_ts = 0, 0, 0, 0
        # if not info["obs_missing_data"] and raw_act_received_flag:
        if not info["obs_missing_data"]:
            # 6. Calculate reward
            reward_range_0_pos_1, reward_range_neg_1_pos_1, \
            reward_range_0_pos_2, reward_range_neg_2_pos_2, \
            reward_range_0_pos_10, reward_range_0_pos_100 = self._calculate_handcrafted_reward(self.obs, action, new_obs, info)
            info['reward_range_0_pos_1'], info['reward_range_neg_1_pos_1'], \
            info['reward_range_0_pos_2'], info['reward_range_neg_2_pos_2'], \
            info['reward_range_0_pos_10'], info[
                'reward_range_0_pos_100'] = reward_range_0_pos_1, reward_range_neg_1_pos_1, reward_range_0_pos_2, reward_range_neg_2_pos_2, reward_range_0_pos_10, reward_range_0_pos_100

            if self.hc_reward_range == 'reward_range_0_pos_1':
                extl_rew = reward_range_0_pos_1
            elif self.hc_reward_range == 'reward_range_neg_1_pos_1':
                extl_rew = reward_range_neg_1_pos_1
            elif self.hc_reward_range == 'reward_range_0_pos_2':
                extl_rew = reward_range_0_pos_2
            elif self.hc_reward_range == 'reward_range_neg_2_pos_2':
                extl_rew = reward_range_neg_2_pos_2
            elif self.hc_reward_range == 'reward_range_0_pos_10':
                extl_rew = reward_range_0_pos_10
            elif self.hc_reward_range == 'reward_range_0_pos_100':
                extl_rew = reward_range_0_pos_100
            else:
                raise ValueError("Wrong reward_range: {}".format(self.hc_reward_range))
            if self.rew_comp is None:
                rew = extl_rew
            else:
                # Note: when calculate immediate reward, keep the input format (1, dim) for MLP or (1, mem_len, dim) for LSTM.
                if self.rew_comp.reward_comp_type == 'MLP':
                    rew = self.rew_comp(self.obs.reshape(1, -1), action.reshape(1, -1), new_obs.reshape(1, -1))
                elif self.rew_comp.reward_comp_type == 'LSTM':
                    mem_end_id = len(self.obs_traj) - 1
                    mem_start_id = max(0, mem_end_id - self.rew_comp.reward_mem_length + 1)
                    # Stack trajectory with the format for obs (1, mem_len, obs_dim)
                    rew = self.rew_comp(np.stack(self.obs_traj[mem_start_id:mem_end_id + 1], axis=1),
                                        np.stack(self.act_traj[mem_start_id:mem_end_id + 1], axis=1),
                                        np.stack(self.obs2_traj[mem_start_id:mem_end_id + 1], axis=1), mem_len=[mem_end_id - mem_start_id + 1])
                else:
                    raise ValueError('Wrong reward_comp_type: {}'.format(self.rew_comp.reward_comp_type))
            rew_ts = datetime.timestamp(datetime.now())  # reward timestamp

            # 7. Determine if task is done
            done = False
            done_ts = datetime.timestamp(datetime.now())  # reward timestamp

            # 7. Save experience to database
            # TODO:
            # self.cloud_db.store_experiences(behaviour_mode,
            #                                            list(obs), obs_time, list(act), act_time,
            #                                            rew, list(new_obs), new_obs_time, datetime.now())
        else:
            raise ValueError("Missing data!")

        # Crucial note: Update current observation after reward computation.
        self.obs, self.proprio_and_extero_obs, self.info = new_obs, new_proprio_and_extero_obs, info
        # Store extl_env to info for diagnostic purpose
        info['extl_rew'] = extl_rew
        info['orig_rew'] = extl_rew
        info['act_datetime'] = act_ts
        info['obs_datetime'] = new_obs_ts

        return new_obs, rew, done, info

    # def _excitor_elicited_raw_act_collection(self, sample_size=30, act_interval=0.5):
    #     """
    #     Function used to collect raw actions elicited by excitor.
    #     :param sample_size: the the number of action is going to collect
    #     :param act_interval: the interval between two sampled actions
    #     :return:
    #     """
    #     excitor_raw_act_source_server = self.internal_env_config['action_space']['raw_act_space']['LAS_for_action_execution']
    #     excitor_raw_act_trajectory = []
    #     sample_i = 0
    #     while sample_i < sample_size:
    #         time.sleep(act_interval)  # Adjust time for excitor raw action collection frequency
    #         # Start receiving and then pause
    #         self.comm_manager.empty_excitor_raw_act_buffer()
    #         self.comm_manager.serve_excitor_raw_act_server(serve_time = 0.1)
    #
    #         # Construct raw action elicited by excitor
    #         excitor_raw_act = []
    #         for device_group in self.raw_act_space_dict:
    #             for device_node_id in self.raw_act_space_dict[device_group]:
    #                 # Only consider node with actuators and construct message in format: "pin float pin float ..."
    #                 if len(self.raw_act_space_dict[device_group][device_node_id]) != 0:
    #                     for device_name in self.raw_act_space_dict[device_group][device_node_id]:
    #                         _, _, device, device_num, device_pin = device_name.split('_')
    #                         device_id = '{}_{}_{}_{}'.format(device_node_id, device, device_num, device_pin)
    #                         raw_act = 0
    #                         if self.comm_manager.excitor_raw_act_buffer[excitor_raw_act_source_server][device][
    #                             device_id].empty():
    #                             # if no raw action, action = 0
    #                             # print("Empty: {}".format(device_id))
    #                             pass
    #                         else:
    #                             # Get the last action and empty other
    #                             while not \
    #                                     self.comm_manager.excitor_raw_act_buffer[excitor_raw_act_source_server][device][
    #                                         device_id].empty():
    #                                 raw_act = \
    #                                 self.comm_manager.excitor_raw_act_buffer[excitor_raw_act_source_server][device][
    #                                     device_id].get()
    #                         # Transform from [0,1] to [-1,1]
    #                         raw_act = raw_act * (self.act_val_max - self.act_val_min) + self.act_val_min
    #                         excitor_raw_act.append(raw_act)
    #         # Add act to sampled trajectory
    #         excitor_raw_act_trajectory.append(excitor_raw_act)
    #         sample_i += 1
    #
    #     excitor_raw_act_trajectory = np.asarray(excitor_raw_act_trajectory)
    #     return excitor_raw_act_trajectory
    #
    # def collect_excitor_based_experiences(self, sample_size=30, act_interval=1):
    #     """
    #     This function is used for collecting excitor based experiences
    #     :param sample_size: the number of samples going to collect
    #     :param act_interval: interval of actions when collecting from excitor
    #     :return:
    #     """
    #     behaviour_mode = "Excitor"
    #     # 1. Enable excitor
    #     self.comm_manager.excitor_client.send_message("/Excitor/Enable", "true")
    #
    #     # 2 Collect actions elicited by excitor.
    #     excitor_raw_act_trajectory = self._excitor_elicited_raw_act_collection(sample_size=sample_size,
    #                                                                            act_interval=act_interval)
    #     # If no actuator is activated for the whole trajectory, it's probably the excitor is unenabled.
    #     if not np.any(excitor_raw_act_trajectory > 0):
    #         return excitor_raw_act_trajectory
    #         raise ValueError(
    #             "Something wrong with excitor_elicited_raw_act_collection! Please check if Excitor is enabled?")
    #
    #     # 3. Disable excitor
    #     self.comm_manager.excitor_client.send_message("/Excitor/Enable", "false")
    #
    #     # 4. Start recording video
    #     self.comm_manager.video_capture_client.send_message("/Video_Capture", "Start")
    #
    #     # 5. Collect (action, observation) trajectories
    #     #    Get initial observation
    #     print("Start collecting experiences...")
    #     while True:
    #         print("Get initial observation")
    #         obs, info = self._collect_and_construct_obs()
    #         obs_time = datetime.now()
    #         if not info['obs_missing_data']:
    #             break
    #         else:
    #             print("info['obs_missing_data']={}".format(info['obs_missing_data']))
    #     print("Collecting experiences...")
    #     excitor_i = 0
    #     while excitor_i < len(excitor_raw_act_trajectory):
    #         act = excitor_raw_act_trajectory[excitor_i]
    #
    #         # 5.1. Start act_confirm_server
    #         self.comm_manager.start_act_confirm_server()
    #         # 5.2. Execute action
    #         self._execute_raw_action(act)
    #         act_time = datetime.now()
    #         # 5.3. Collect and construct observation
    #         new_obs, info = self._collect_and_construct_obs()
    #         new_obs_time = datetime.now()
    #         # 5.4. Collect action confirmation after observation to earn enough time for receiving them.
    #         self.comm_manager.pause_act_confirm_server()
    #         raw_act_received_flag, raw_act_confirm = self._collect_raw_action_execution_confirmation()
    #         info["raw_act_received_flag"] = raw_act_received_flag
    #         info["raw_act_confirm"] = raw_act_confirm
    #
    #         # 5.5 Store (action, observation) trajectory if no data missing
    #         if not info['obs_missing_data'] and info["raw_act_received_flag"]:
    #             self.local_db.store_experience(behaviour_mode,
    #                                            list(obs), obs_time, list(act), act_time,
    #                                            0, list(new_obs), new_obs_time)
    #             obs = new_obs
    #             obs_time = new_obs_time
    #             excitor_i += 1
    #     # 6. Stop recording video
    #     self.comm_manager.video_capture_client.send_message("/Video_Capture", "Stop")
    #     time.sleep(5)  # Add an idle time after stopping video capture to allow video encoding completion.
    #     return excitor_raw_act_trajectory
    #
    # def collect_random_experiences(self, sample_size=30):
    #     behaviour_mode = "Random"
    #     act_trajectory = []
    #     # 1. Disable excitor
    #     self.comm_manager.excitor_client.send_message("/Excitor/Enable", "false")
    #
    #     # 2. Start recording video
    #     self.comm_manager.video_capture_client.send_message("/Video_Capture", "Start")
    #
    #     # 3. Collect (action, observation) trajectories
    #     #    Get initial observation
    #     print("Start collecting experiences...")
    #     while True:
    #         print("Get initial observation")
    #         obs, info = self._collect_and_construct_obs()
    #         obs_time = datetime.now()
    #         if not info['obs_missing_data']:
    #             break
    #         else:
    #             print("info['obs_missing_data']={}".format(info['obs_missing_data']))
    #     random_act_i = 0
    #     while random_act_i < sample_size:
    #         act = np.random.uniform(-1, 1, self.raw_act_space_dim)
    #         # 3.1. Start act_confirm_server
    #         self.comm_manager.start_act_confirm_server()
    #         # 3.2. Execute action
    #         self._execute_raw_action(act)
    #         act_time = datetime.now()
    #         # 3.3. Collect and construct observation
    #         new_obs, info = self._collect_and_construct_obs()
    #         new_obs_time = datetime.now()
    #         # 3.4. Collect action confirmation after observation to earn enough time for receiving them.
    #         self.comm_manager.pause_act_confirm_server()
    #         raw_act_received_flag, raw_act_confirm = self._collect_raw_action_execution_confirmation()
    #         info["raw_act_received_flag"] = raw_act_received_flag
    #         info["raw_act_confirm"] = raw_act_confirm
    #
    #         # 3.5 Store (action, observation) trajectory if no data missing
    #         # if not info['obs_missing_data'] and info["raw_act_received_flag"]:
    #         if not info['obs_missing_data']:
    #             act_trajectory.append(act)
    #             # proprio_obs_active = len(np.where(np.array(info['proprio_obs_list']) != 0)[0])
    #             # act_active = len(np.where(act != -1)[0]) * self.internal_env_config['obs_space']['proprioception'][
    #             #     'obs_frequency']
    #             # print("\t act_active:{}".format(act_active))
    #             # print("\t proprio_obs_active:{}".format(proprio_obs_active))
    #             # behaviour_mode, obs, obs_time, act, act_time, rew, obs2, obs2_time, create_time
    #             self.local_db.store_experience(behaviour_mode,
    #                                            list(obs), obs_time, list(act), act_time,
    #                                            0, list(new_obs), new_obs_time)
    #             obs = new_obs
    #             obs_time = new_obs_time
    #             random_act_i += 1
    #         else:
    #             print("info['obs_missing_data']={}, info['raw_act_received_flag']={}".format(info['obs_missing_data'],
    #                                                                                          info["raw_act_received_flag"]))
    #     # 4. Stop recording video
    #     self.comm_manager.video_capture_client.send_message("/Video_Capture", "Stop")
    #     time.sleep(5)  # Add an idle time after stopping video capture to allow video encoding completion.
    #     return np.array(act_trajectory)
    #
    # def collect_silence_experiences(self, sample_size=30):
    #     behaviour_mode = "Silence"
    #     act_trajectory = []
    #     # 1. Disable excitor
    #     self.comm_manager.excitor_client.send_message("/Excitor/Enable", "false")
    #
    #     # 2. Start recording video
    #     self.comm_manager.video_capture_client.send_message("/Video_Capture", "Start")
    #
    #     # 3. Collect (action, observation) trajectories
    #     #    Get initial observation
    #     print("Start collecting experiences...")
    #     while True:
    #         print("Get initial observation")
    #         obs, info = self._collect_and_construct_obs()
    #         obs_time = datetime.now()
    #         if not info['obs_missing_data']:
    #             break
    #         else:
    #             print("info['obs_missing_data']={}".format(info['obs_missing_data']))
    #     random_act_i = 0
    #     while random_act_i < sample_size:
    #         act = np.random.uniform(-1, 0, self.raw_act_space_dim)
    #         # 3.1. Start act_confirm_server
    #         self.comm_manager.start_act_confirm_server()
    #         # 3.2. Execute action
    #         self._execute_raw_action(act)
    #         act_time = datetime.now()
    #         # 3.3. Collect and construct observation
    #         new_obs, info = self._collect_and_construct_obs()
    #         new_obs_time = datetime.now()
    #         # 3.4. Collect action confirmation after observation to earn enough time for receiving them.
    #         self.comm_manager.pause_act_confirm_server()
    #         raw_act_received_flag, raw_act_confirm = self._collect_raw_action_execution_confirmation()
    #         info["raw_act_received_flag"] = raw_act_received_flag
    #         info["raw_act_confirm"] = raw_act_confirm
    #
    #         # 3.5 Store (action, observation) trajectory if no data missing
    #         # if not info['obs_missing_data'] and info["raw_act_received_flag"]:
    #         if not info['obs_missing_data']:
    #             act_trajectory.append(act)
    #             # proprio_obs_active = len(np.where(np.array(info['proprio_obs_list']) != 0)[0])
    #             # act_active = len(np.where(act != -1)[0]) * self.internal_env_config['obs_space']['proprioception'][
    #             #     'obs_frequency']
    #             # print("\t act_active:{}".format(act_active))
    #             # print("\t proprio_obs_active:{}".format(proprio_obs_active))
    #             # behaviour_mode, obs, obs_time, act, act_time, rew, obs2, obs2_time, create_time
    #             self.local_db.store_experience(behaviour_mode,
    #                                            list(obs), obs_time, list(act), act_time,
    #                                            0, list(new_obs), new_obs_time)
    #             obs = new_obs
    #             obs_time = new_obs_time
    #             random_act_i += 1
    #         else:
    #             print("info['obs_missing_data']={}, info['raw_act_received_flag']={}".format(info['obs_missing_data'],
    #                                                                                          info["raw_act_received_flag"]))
    #     # 4. Stop recording video
    #     self.comm_manager.video_capture_client.send_message("/Video_Capture", "Stop")
    #     time.sleep(5)  # Add an idle time after stopping video capture to allow video encoding completion.
    #     return np.array(act_trajectory)

def main():
    from pl import las_config
    int_env = LASIntlEnv(las_config)
    import pdb; pdb.set_trace()
    start_time = time.time()
    # int_env.comm_manager.serve_excitor_raw_act_server(serve_time=0.01)
    int_env.comm_manager.serve_obs_server(serve_time=0.1)
    print("{}s".format(time.time()-start_time))


if __name__ == "__main__":
    main()



