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
from numpy import dot
from numpy.linalg import norm

class ReplayBuffer:

    def __init__(self,obs_dim:int,size:int,batch_size:int):

        self.obs_buf=np.zeros([size,obs_dim],dtype=np.float32)
        self.next_obs_buf=np.zeros([size,obs_dim],dtype=np.float32)
        self.acts_buf=np.zeros([size],dtype=np.float32)
        self.rews_buf=np.zeros([size],dtype=np.float32)
        self.done_buf=np.zeros(size,dtype=np.float32)
        self.epi_rew_buf=np.zeros([size],dtype=np.float32)

        self.max_size,self.batch_size=size,batch_size
        self.ptr,self.size=0,0


    def store(self,obs:np.ndarray,act:np.ndarray,rew:float,next_obs:np.ndarray,done:bool,epi:float):

        done=0. if done else 1

        self.obs_buf[self.ptr]=obs
        self.next_obs_buf[self.ptr]=next_obs
        self.acts_buf[self.ptr]=act
        self.rews_buf[self.ptr]=rew
        self.done_buf[self.ptr]=done
        self.epi_rew_buf[self.ptr]=epi

        self.ptr=(self.ptr+1)%self.max_size
        self.size=min(self.size+1,self.max_size)

    def cos_sim(self,b1,b2):

        return dot(b1,b2)/(norm(b1)*norm(b2))

    def __len__(self):

        return self.size


    def extract_sim_state(self,idx,lastest_sample):

        states=self.obs_buf[idx]

        sim_array=list()

        for state in states:

            sim=self.cos_sim(lastest_sample[0],state)
            sim_array.append(sim)

        sim_array=np.array(sim_array,dtype=np.float32)
        arg_index=np.argsort(sim_array)
        idx=np.where(arg_index>=self.batch_size-32)

        obs=self.obs_buf[idx]
        acts=self.acts_buf[idx]
        rews=self.rews_buf[idx]
        next_obs=self.next_obs_buf[idx]
        done=self.done_buf[idx]
        epi_rews=self.epi_rew_buf[idx]
        arg_index=np.argsort(epi_rews)
        idx=np.where(arg_index>=28)\

        obs=obs[idx]
        acts=acts[idx]
        rews=rews[idx]
        next_obs=next_obs[idx]
        done=done[idx]


        return obs,acts,rews,next_obs,done


    def sample_batch(self,lastest_sample) -> Dict[str,np.ndarray]:

        assert len(self)>=2*self.batch_size

        idx0=np.random.choice(self.size,self.batch_size,replace=True)



        obs1,acts1,rews1,next_obs1,done1=self.extract_sim_state(idx0,lastest_sample)



        idx1=np.random.choice(self.size,self.batch_size,replace=True)
        idx2=np.random.choice(self.size,self.batch_size,replace=True)




        idx=np.hstack([idx1,idx2])
        inter_epi_rews=self.epi_rew_buf[idx]
        inter_obs=self.obs_buf[idx]
        inter_next_obs=self.next_obs_buf[idx]
        inter_acts=self.acts_buf[idx]
        inter_done=self.done_buf[idx]
        inter_rews=self.rews_buf[idx]

        arg_index=np.argsort(inter_epi_rews)
        idx=np.where(arg_index>=self.batch_size+4)


        obs2=inter_obs[idx]
        acts2=inter_acts[idx]
        rews2=inter_rews[idx]
        next_obs2=inter_obs[idx]
        done2=inter_done[idx]
        epi_rews2=inter_epi_rews[idx]

        obs=np.concatenate((obs1,obs2),axis=0)
        next_obs=np.concatenate((next_obs1,next_obs2),axis=0)
        acts=np.concatenate((acts1,acts2),axis=None)
        rews=np.concatenate((rews1,rews2),axis=None)
        done=np.concatenate((done1,done2),axis=None)


        return dict(obs=obs,acts=acts,rews=rews,next_obs=next_obs,done=done)

class MuNet(nn.Module):
    def __init__(self):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc3(x)) * 2

        return mu

class QNet(nn.Module):

    def __init__(self, ):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(3, 64)
        self.fc_a = nn.Linear(1, 64)
        self.fc_q = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32, 1)

    def forward(self, s, a):
        h1 = F.relu(self.fc_s(s))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=1)
        q = F.relu(self.fc_q(cat))

        return self.fc_out(q)

class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)

        self.x_prev = x
        return x

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True



class DDPGAgent:

    def __init__(self,env:gym.Env,memory_size:int,batch_size:int,gamma:float=0.99):

        obs_dim=3

        self.env=env
        self.memory=ReplayBuffer(obs_dim,memory_size,batch_size)
        self.batch_size=batch_size
        self.gamma=gamma

        self.device=torch.device("cuda")
        print(self.device)

        self.q,self.q_target=QNet().to(self.device),QNet().to(self.device)
        self.mu,self.mu_target=MuNet().to(self.device),MuNet().to(self.device)

        self.q_target.load_state_dict(self.q.state_dict())
        self.mu_target.load_state_dict(self.mu.state_dict())

        self.q_target.eval()
        self.mu_target.eval()

        self.mu_optimizer=optim.Adam(self.mu.parameters(),lr=0.0005)
        self.q_optimizer=optim.Adam(self.q.parameters(),lr=0.001)

        self.is_test=False

        self.ou_noise=OrnsteinUhlenbeckNoise(mu=np.zeros(1))


    def select_action(self,state:np.ndarray) -> np.ndarray:

        selected_action=self.mu(torch.FloatTensor(state).to(self.device))
        selected_action=selected_action.detach().cpu().numpy()+self.ou_noise()

        return selected_action


    def step(self,action:np.ndarray) ->Tuple[np.ndarray,np.float64,bool]:

        next_state,reward,done,_=self.env.step(action)

        return next_state,reward,done

    def update_model(self,lastest_transition:List):

        samples=self.memory.sample_batch(lastest_transition)

        device=self.device
        state=torch.FloatTensor(samples['obs']).to(device)
        next_state=torch.FloatTensor(samples['next_obs']).to(device)
        action=torch.FloatTensor(samples['acts']).reshape(-1,1).to(device)
        reward=torch.FloatTensor(samples['rews']).reshape(-1,1).to(device)
        done_mask=torch.FloatTensor(samples['done']).reshape(-1,1).to(device)

        target = (reward + self.gamma * self.q_target(next_state, self.mu_target(next_state)) * done_mask).to(
            self.device)

        q_loss = F.smooth_l1_loss(self.q(state, action), target.detach())
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        mu_loss = -self.q(state, self.mu(state)).mean()
        self.mu_optimizer.zero_grad()
        mu_loss.backward()
        self.mu_optimizer.step()

    def soft_update(self, net, net_target):
        tau = 0.005
        for param_target, param in zip(net_target.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1. - tau) + tau * param.data)



    def train(self,num_frames:int) ->np.ndarray:

        self.is_test=False
        reward_lst=list()
        state=self.env.reset()
        score=0
        episode=list()
        result_score=0


        for frame_idx in range(num_frames):

            episode=list()
            score=0
            done=False
            state=self.env.reset()
            while not done:

                action=self.select_action(state)
                next_state,reward,done=self.step(action)
                transition=[state,action,reward/100.,next_state,done]
                state=next_state
                result_score+=reward
                score+=reward
                episode.append(transition)



            lastest_transition=episode[-1]

            if (frame_idx+1)%5==0:
                reward_lst.append(result_score/5)
                result_score=0

            if len(self.memory)>=self.batch_size:
                self.update_model(lastest_transition)
                self.soft_update(self.mu,self.mu_target)
                self.soft_update(self.q,self.q_target)


            for transition in episode:
                transition.append(score)
                self.memory.store(*transition)

            episode=list()
            score=0

        self.env.close()
        reward_lst=np.array(reward_lst,dtype=np.float32)
        return reward_lst







