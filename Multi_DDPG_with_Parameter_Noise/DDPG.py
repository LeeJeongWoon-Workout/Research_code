import random

GAMMA=0.99
BATCH_SIZE=64
import torch
from Network import QNet,MuNet
from utils import OrnsteinUhlenbeckProcess,AdaptiveParamNoiseSpec,hard_update,weightSync
from ReplayBuffer import ReplayBuffer
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np

class DDPG:

    def __init__(self,num_episode,q_lr=1e-3,a_lr=1e-4,gamma=GAMMA,batch_size=BATCH_SIZE):

        self.gamma=GAMMA
        self.batch_size=BATCH_SIZE
        self.device=torch.device("cuda")

        self.actor=MuNet().to(self.device)
        self.actor_perturbed=MuNet().to(self.device)
        self.actor_target=MuNet().to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.eval()

        self.critic=QNet().to(self.device)
        self.critic_target=QNet().to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()

        self.optimizer_actor=torch.optim.Adam(self.actor.parameters(),lr=a_lr)
        self.optimizer_critic=torch.optim.Adam(self.critic.parameters(),lr=q_lr)

        self.critic_loss=nn.MSELoss()

        self.noise=OrnsteinUhlenbeckProcess(dimension=1,num_steps=num_episode)

    def action(self,s,noise,para):
        obs=torch.FloatTensor(s).to(self.device)

        self.actor.eval()
        self.actor_perturbed.eval()

        if para is not None:
            a=self.actor_perturbed(obs).detach().cpu().numpy()
        else:
            a=self.actor(obs).detach().cpu().numpy()

        self.actor.train()

        if noise is not None:
            a=a+noise

        return a


    def perturb_actor_parameters(self, param_noise):
        """Apply parameter noise to actor model, for exploration"""
        hard_update(self.actor_perturbed, self.actor)
        params = self.actor_perturbed.state_dict()
        for name in params:
            if 'ln' in name:
                pass
            param = params[name]
            random = torch.randn(param.shape).to(self.device)

            param += random * param_noise.current_stddev



    def train(self,memory:ReplayBuffer,hmemory:ReplayBuffer,batch_size):
        tau=0.25
        r=random.uniform(0,1)

        if tau>r:
            training_data=hmemory.sample(batch_size)

        else:
            training_data=memory.sample(batch_size)

        batch_s, batch_a, batch_r, batch_s1, batch_done = zip(*training_data)
        s1 = torch.FloatTensor(batch_s).to(self.device)
        a1 = torch.FloatTensor(batch_a).to(self.device)
        r1 = torch.FloatTensor(np.array(batch_r).reshape(-1, 1)).to(self.device)
        s2 = torch.FloatTensor(batch_s1).to(self.device)
        d = torch.FloatTensor(1.0 * np.array(batch_done).reshape(-1, 1)).to(self.device)

        a2 = self.actor_target(s2)
        # ---------------------- optimize critic ----------------------
        next_val = self.critic_target(s2, a2).detach()
        q_expected = r1 + self.gamma * next_val * (1.0 - d)
        q_predicted = self.critic(s1, a1)

        # compute critic loss, and update the critic
        loss_critic = self.critic_loss(q_predicted, q_expected)
        self.optimizer_critic.zero_grad()
        loss_critic.backward()
        self.optimizer_critic.step()

        # ---------------------- optimize actor ----------------------
        pred_a1 = self.actor.forward(s1)
        loss_actor = -1 * self.critic(s1, pred_a1)
        loss_actor = loss_actor.mean()

        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        self.optimizer_actor.step()

        # sychronize target network with fast moving one
        weightSync(self.critic_target, self.critic)
        weightSync(self.actor_target, self.actor)




