import numpy as np
import torch
import torch.nn as nn
from math import sqrt
import torch.nn.functional as F
#environment
import gym

def weightSync(target_model,source_model,tau=1e-3):
    for parameter_target,parameter_source in zip(target_model.parameters(),source_model.parameters()):
        parameter_target.data.copy_((1-tau)*parameter_target.data+tau*parameter_source.data)

class OrnsteinUhlenbeckProcess:
    def __init__(self,num_steps,mu=np.zeros(1),sigma=0.05,theta=.25,dimension=1e-2,x0=None):
        self.theta=theta
        self.mu=mu
        self.sigma=sigma
        self.dt=dimension
        self.x0=x0
        self.reset()

    def step(self):
        x=self.x_prev+self.theta*(self.mu-self.x_prev)*self.dt+\
        self.sigma*np.sqrt(self.dt)*np.random.normal(size=self.mu.shape)

        self.x_prev=x
        return x

    def reset(self):
        self.x_prev=self.x0 if self.x0 else np.zeros_like(self.mu)


class AdaptiveParamNoiseSpec(object):

    def __init__(self,initial_stddev=0.1,desired_action_stddev=0.2,adaptation_coefficient=1.01):

        self.initial_stddev=initial_stddev
        self.desired_action_stddev=desired_action_stddev
        self.adaptation_coefficient=adaptation_coefficient

        self.current_stddev=initial_stddev

    def adapt(self,distance):
        if distance>self.desired_action_stddev:

            self.current_stddev/=self.adaptation_coefficient
        else:
            self.current_stddev*=self.adaptation_coefficient

    def get_stats(self):
        stats={
            'param_noise_stddev':self.current_stddev,
        }
        return stats

    def __repr__(self):
        fmt = 'AdaptiveParamNoiseSpec(initial_stddev={}, desired_action_stddev={}, adaptation_coefficient={})'
        return fmt.format(self.initial_stddev, self.desired_action_stddev, self.adaptation_coefficient)

def ddpg_distance_metric(actions1,actions2):
    diff=actions1-actions2
    mean_diff=np.mean(np.square(diff),axis=0)
    dist=sqrt(np.mean(mean_diff))
    return dist


def hard_update(target,source):
    for target_param,param in zip(target.parameters(),source.parameters()):
        target_param.data.copy_(param.data)

