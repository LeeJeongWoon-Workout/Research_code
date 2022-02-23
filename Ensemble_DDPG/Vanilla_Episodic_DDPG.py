import os
from typing import Dict, List, Tuple
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import random
from IPython.display import clear_output


from Ensemble_DDPG.ReplayBuffer import ReplayBuffer
from Ensemble_DDPG.Noisy import OrnsteinUhlenbeckNoise
from Ensemble_DDPG.Network import QNet,MuNet

class DDPGAgent_epi:

    def __init__(self,env:gym.Env,memory_size:int,batch_size:int,tau:float,gamma:float=0.99):

        obs_dim=3

        self.env=env
        self.memory=ReplayBuffer(obs_dim,memory_size,batch_size)
        self.hmemory=ReplayBuffer(obs_dim,memory_size,batch_size)
        self.batch_size=batch_size
        self.gamma=gamma
        self.tau=tau


        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(self.device)

        self.q,self.q_target=QNet().to(self.device),QNet().to(self.device)
        self.mu,self.mu_target=MuNet().to(self.device),MuNet().to(self.device)

        self.q_target.load_state_dict(self.q.state_dict())
        self.mu_target.load_state_dict(self.mu.state_dict())

        self.q_target.eval()
        self.mu_target.eval()

        self.mu_optimizer=optim.Adam(self.mu.parameters(),lr=0.0005)
        self.q_optimizer=optim.Adam(self.q.parameters(),lr=0.001)


        self.transition=list()
        self.is_test=False

        self.ou_noise=OrnsteinUhlenbeckNoise(mu=np.zeros(1))


    def select_action(self,state:np.ndarray) ->np.ndarray:

        selected_action=self.mu(torch.FloatTensor(state).to(self.device))
        selected_action=selected_action.detach().cpu().numpy()+self.ou_noise()

        if not self.is_test:
            self.transition=[state,selected_action]

        return selected_action

    def step(self,action:np.ndarray) ->Tuple[np.ndarray,np.float32,bool]:

        next_state,reward,done,_=self.env.step(action)

        if not self.is_test:
            self.transition+=[reward/100.,next_state,done]
            self.memory.store(*self.transition)


        return next_state,reward,done


    def update_model(self):

        th=self.tau
        ran=random.uniform(0,1)

        if th>ran:
            samples=self.hmemory.sample_batch()

        else:
            samples=self.memory.sample_batch()


        device=self.device
        state=torch.FloatTensor(samples['obs']).to(device)
        next_state=torch.FloatTensor(samples['next_obs']).to(device)
        action=torch.FloatTensor(samples['acts'].reshape(-1,1)).to(device)
        reward=torch.FloatTensor(samples['rews'].reshape(-1,1)).to(device)
        done_mask=torch.FloatTensor(samples['done'].reshape(-1,1)).to(device)


        target=(reward+self.gamma*self.q_target(next_state,self.mu_target(next_state))*done_mask).to(self.device)

        q_loss=F.smooth_l1_loss(self.q(state,action),target.detach())
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        mu_loss=-self.q(state,self.mu(state)).mean()
        self.mu_optimizer.zero_grad()
        mu_loss.backward()
        self.mu_optimizer.step()


    def soft_update(self,net, net_target):
        tau=0.005
        for param_target, param in zip(net_target.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1. - tau) + tau * param.data)


    def train(self,num_frames:int) -> np.ndarray:

        self.is_test=False
        reward_lst=list()
        episode = list()
        state=self.env.reset()
        R_max=-10000000.
        score=0
        result_score = 0


        for frame_idx in range(num_frames):
            done=False
            state=self.env.reset()
            while not done:
                action=self.select_action(state)
                next_state,reward,done=self.step(action)
                transition=[state,action,reward/100.,next_state,done]
                state=next_state
                score+=reward
                result_score+=reward
                episode.append(transition)


            if score>R_max:
                R_max=score
                for transition in episode:
                    self.hmemory.store(*transition)

            episode=list()
            score=0

            if (frame_idx+1)%10==0:
                reward_lst.append(result_score/10)
                result_score=0

            if (len(self.memory)>=self.batch_size and len(self.hmemory)>=self.batch_size) and len(self.memory)>=1000:

                self.update_model()
                self.soft_update(self.mu,self.mu_target)
                self.soft_update(self.q,self.q_target)

        self.env.close()
        return reward_lst



def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True



