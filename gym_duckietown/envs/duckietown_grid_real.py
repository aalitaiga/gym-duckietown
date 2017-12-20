from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import curses
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
from pycolab import ascii_art
from pycolab import human_ui
from pycolab.prefab_parts import sprites as prefab_sprites
import visdom



action_orientation = {
    0: prefab_sprites.MazeWalker._NORTH,
    1: prefab_sprites.MazeWalker._SOUTH,
    2: prefab_sprites.MazeWalker._WEST,
    3: prefab_sprites.MazeWalker._EAST,
}


def recvArray(socket):
    """Receive a numpy array over zmq"""
    md = socket.recv_json()
    msg = socket.recv(copy=True, track=False)
    buf = buffer(msg)
    A = numpy.frombuffer(buf, dtype=md['dtype'])
    A = A.reshape(md['shape'])
    return A



class DuckietownGridReal(gym.Env):

    SERVER_PORT=7777
    SERVER_ADDR="localhost"
    def __init__(self,
                 serverAddr=SERVER_ADDR,
                 serverPort=SERVER_PORT):

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiBinary(3)

        # Connect to the Gym bridge ROS node
        print("connecting...")
        context = zmq.Context()
        self.socket = context.socket(zmq.PAIR)
        self.socket.connect("tcp://%s:%s" % (serverAddr, serverPort))
        print("connected! :)")
        self.reset()

        
    def _step(self, action):
        # Send the action to the zmq/ROS bridge
        self.socket.send_json({
            "command":"action",
            "values": [ float(action) ]
        })

        # we don't care about rewards for now - just trying to deploy on the robot
        reward = 0

        # wait for an observation
        observation = self.get_observation()

        return np.array(observation), reward, False, ""

    def _reset(self):
        self.socket.send_json({
            "command":"reset"
        })

        observation = self.get_observation()
        return observation

    def _render(self, mode="human", close=False):
        # raise NotImplementedError
        pass

    # this needs to be blocking (ie wait until we get something) I should verify that it is
    def get_observation(self):
        return recvArray(self.socket)
