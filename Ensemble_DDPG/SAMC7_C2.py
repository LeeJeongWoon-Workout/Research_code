from typing import Tuple
import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random

from ReplayBuffer import ReplayBuffer
from Noisy import OrnsteinUhlenbeckNoise
from Network import QNet,MuNet

#paper review: https://velog.io/@everyman123/Ensemble-Bootstrapped-Deep-Deterministic-Policy-Gradient-for-vision-based-Robotic-Grasping-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0
#paper link: https://ieeexplore.ieee.org/document/9316755


class SAMC7_C2:

    def __init__(self, env: gym.Env, memory_size: int, batch_size: int, tau: float = 0.25, choice: int = 4,
                 gamma: float = 0.99):

        obs_dim=3

        self.env=env
        self.memory=ReplayBuffer(obs_dim,memory_size,batch_size)
        self.hmemory=ReplayBuffer(obs_dim,memory_size,batch_size)
        self.batch_size=batch_size
        self.gamma=gamma
        self.tau=tau
        self.choice=choice
        self.device=torch.device("cuda")

        print(self.device)
        self.mu,self.mu_target=MuNet().to(self.device),MuNet().to(self.device)
        self.mu_target.load_state_dict(self.mu.state_dict())

        self.mu_target.eval()
        self.mu_optimizer=optim.Adam(self.mu.parameters(),lr=0.0005)

        self.q_lst=list()

        for i in range(7):
            self.q,self.q_target=QNet().to(self.device),QNet().to(self.device)
            self.q_target.load_state_dict(self.q.state_dict())
            self.q_target.eval()
            self.q_optim=optim.Adam(self.q.parameters(),lr=0.01)

            self.q_lst.append((self.q,self.q_target,self.q_optim))


        print('critics number: ',len(self.q_lst))
        self.is_test=False
        self.ou_noise=OrnsteinUhlenbeckNoise(mu=np.zeros(1))

    def select_action(self, state: np.ndarray) -> np.ndarray:

        selected_action = self.mu(torch.FloatTensor(state).to(self.device))
        selected_action = selected_action.detach().cpu().numpy() + self.ou_noise()

        if not self.is_test:
            self.transition = [state, selected_action]

        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float32, bool]:

        next_state, reward, done, _ = self.env.step(action)

        if not self.is_test:
            self.transition += [reward / 100., next_state, done]
            self.memory.store(*self.transition)

        return next_state, reward, done

    def soft_update(self, net, net_target):
        tau = 0.005
        for param_target, param in zip(net_target.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1. - tau) + tau * param.data)


    def train(self,num_frames:int,a:float,b:float,w:float) -> np.ndarray:

        self.is_test=False
        reward_lst=list()
        episode=list()
        state=self.env.reset()
        R_max=-10000
        score=0
        result_score=0

        for frame_idx in range(num_frames):
            done=False
            state=self.env.reset()
            steps=0

            while not done:
                action=self.select_action(state)
                next_state,reward,done=self.step(action)
                transition=[state,action,reward/100.,next_state,done]
                state=next_state
                score+=reward
                result_score+=reward
                episode.append(transition)
                steps+=1

            if score>R_max:
                R_max=score
                for transition in episode:
                    self.hmemory.store(*transition)

            episdoe=list()
            score=0

            if (len(self.memory) >= self.batch_size and len(self.hmemory) >= self.batch_size) and len(
                    self.memory) >= 1000:
                for i in range(3):
                    self.update_model(a,b,w)

            if (frame_idx+1)%10==0:
                reward_lst.append(result_score/10)
                result_score=0

        self.env.close()
        return reward_lst


    def update_model(self,a,b,w):

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

        (q1,q1_target,q1_optim),(q2,q2_target,q2_optim)=random.sample(self.q_lst,2)


        #L_avg
        q_target_avg=q1_target(next_state,self.mu_target(next_state))*done_mask
        q_target_avg+=q2_target(next_state,self.mu_target(next_state))*done_mask


        q_target_avg/=2.0

        q_avg=q1(state,action)
        q_avg+=q2(state,action)


        q_avg/=2.0

        TD_target=(reward+self.gamma*q_target_avg).to(device)
        L_avg=F.smooth_l1_loss(q_avg,TD_target.detach())


        # Critic Loss
        target1=(reward+self.gamma*q1_target(next_state,self.mu_target(next_state))*done_mask).to(device)
        target2=(reward+self.gamma*q2_target(next_state,self.mu_target(next_state))*done_mask).to(device)


        q1_loss=a*L_avg.detach()+b*F.smooth_l1_loss(q1(state,action),target1.detach())+w*F.smooth_l1_loss(q1(state,action),q_avg.detach())
        q2_loss=a*L_avg.detach()+b*F.smooth_l1_loss(q2(state,action),target2.detach())+w*F.smooth_l1_loss(q2(state,action),q_avg.detach())





        q1_optim.zero_grad()
        q2_optim.zero_grad()


        q1_loss.backward(retain_graph=True)
        q2_loss.backward()


        q1_optim.step()
        q2_optim.step()



        q_avg=q1(state,action)
        q_avg+=q2(state,action)



        q_avg/=2.0

        mu_loss=-q_avg.mean()
        self.mu_optimizer.zero_grad()
        mu_loss.backward()
        self.mu_optimizer.step()

        self.soft_update(q1,q1_target)
        self.soft_update(q2,q2_target)


