import numpy as np
import random
import os
from typing import Dict,List,Tuple
import gym
import random

env=gym.make('Pendulum-v1')
class ReplayBuffer(object):
    def __init__(self, maxlen=60000):
        self.maxlen = maxlen
        # self.data = deque(maxlen = self.maxlen)
        self.data = []
        self.position = 0
        self.size=0
    #         self.initialize(init_length=1000, envir=env)


    def add(self, ep):
        self.data.append(ep)
        self.position = (self.position + 1) % self.maxlen
        # self.data[self.position] = tuple(ep)
        self.size=min(self.size+1,self.maxlen)
    def sample(self, batch_size):
        return random.sample(self.data, batch_size)

    def __len__(self):
        return self.size
