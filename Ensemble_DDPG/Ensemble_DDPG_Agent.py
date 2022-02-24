from typing import Tuple
import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random

from Ensemble_DDPG.ReplayBuffer import ReplayBuffer
from Ensemble_DDPG.Noisy import OrnsteinUhlenbeckNoise
from Ensemble_DDPG.Network import QNet,MuNet

#paper review: https://velog.io/@everyman123/Ensemble-Bootstrapped-Deep-Deterministic-Policy-Gradient-for-vision-based-Robotic-Grasping-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0
#paper link: https://ieeexplore.ieee.org/document/9316755


class Ensemble_DDPGAgnet1:

        def __init__(self, env: gym.Env, memory_size: int, batch_size: int, tau: float=0.25,choice:int=4,gamma: float = 0.99):
            obs_dim = 3

            self.env = env
            self.memory = ReplayBuffer(obs_dim, memory_size, batch_size)
            self.hmemory = ReplayBuffer(obs_dim, memory_size, batch_size)
            self.batch_size = batch_size
            self.gamma = gamma
            self.tau = tau
            self.choice=choice
            self.device=torch.device("cuda")

            print(self.device)
            self.mu,self.mu_target=MuNet().to(self.device),MuNet().to(self.device)
            self.mu_target.load_state_dict(self.mu.state_dict())
            # detach대신 eval로 학습 정지 target은 절대 backpropogation하지 말아야 한다.
            self.mu_target.eval()
            self.mu_optimizer=optim.Adam(self.mu.parameters(),lr=0.0005)

            self.q_lst=list()

            self.q1,self.q1_target=QNet().to(self.device),QNet().to(self.device)
            self.q1_target.load_state_dict(self.q1.state_dict())
            self.q1_target.eval()
            self.q1_optimizer=optim.Adam(self.q1.parameters(),lr=0.001)
            self.q_lst.append((self.q1,self.q1_target,self.q1_optimizer))

            self.q2,self.q2_target=QNet().to(self.device),QNet().to(self.device)
            self.q2_target.load_state_dict(self.q2.state_dict())
            self.q2_target.eval()
            self.q2_optimizer=optim.Adam(self.q2.parameters(),lr=0.001)
            self.q_lst.append((self.q2,self.q2_target,self.q2_optimizer))

            self.q3,self.q3_target=QNet().to(self.device),QNet().to(self.device)
            self.q3_target.load_state_dict(self.q3.state_dict())
            self.q3_target.eval()
            self.q3_optimizer=optim.Adam(self.q3.parameters(),lr=0.001)
            self.q_lst.append((self.q3,self.q3_target,self.q3_optimizer))

            self.q4,self.q4_target=QNet().to(self.device),QNet().to(self.device)
            self.q4_target.load_state_dict(self.q4.state_dict())
            self.q4_target.eval()
            self.q4_optimizer=optim.Adam(self.q4.parameters(),lr=0.001)
            self.q_lst.append((self.q4,self.q4_target,self.q4_optimizer))

            self.q5,self.q5_target=QNet().to(self.device),QNet().to(self.device)
            self.q5_target.load_state_dict(self.q5.state_dict())
            self.q5_target.eval()
            self.q5_optimizer=optim.Adam(self.q5.parameters(),lr=0.001)
            self.q_lst.append((self.q5,self.q5_target,self.q5_optimizer))

            self.q6,self.q6_target=QNet().to(self.device),QNet().to(self.device)
            self.q6_target.load_state_dict(self.q6.state_dict())
            self.q6_target.eval()
            self.q6_optimizer=optim.Adam(self.q6.parameters(),lr=0.001)
            self.q_lst.append((self.q6,self.q6_target,self.q6_optimizer))

            self.q7,self.q7_target=QNet().to(self.device),QNet().to(self.device)
            self.q7_target.load_state_dict(self.q7.state_dict())
            self.q7_target.eval()
            self.q7_optimizer=optim.Adam(self.q7.parameters(),lr=0.001)
            self.q_lst.append((self.q7,self.q7_target,self.q7_optimizer))



            print("critics numbers: ",len(self.q_lst))

            self.transition=list()
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


        def hard_update(self,net,net_target):

                net.load_state_dict(net_target.state_dict())


        def train(self,num_frames:int)->np.ndarray:

            self.is_test=False
            reward_lst=list()
            episode=list()
            state=self.env.reset()
            R_max=-1000000.
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

                episode=list()
                score=0

                if (len(self.memory) >= self.batch_size and len(self.hmemory) >= self.batch_size) and len(
                        self.memory) >= 1000:
                    for i in range(3):
                        self.update_model()

                if (frame_idx+1)%10==0:
                    reward_lst.append(result_score/10)
                    result_score=0

            self.env.close()
            return reward_lst




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


            #q,q_target,q_optimizer ranomly choose

            (q1,q1_target,q1_optim),(q2,q2_target,q2_optim)=random.sample(self.q_lst,2)


            #L_avg
            q_target_avg=q1_target(next_state,self.mu_target(next_state))*done_mask
            q_target_avg+=q2_target(next_state,self.mu_target(next_state))*done_mask

            q_target_avg/=2.0

            q_avg=q1(state,action)
            q_avg+=q2(state,action)

            q_avg/=2.0

            TD_target=(reward+self.gamma*q_target_avg).to(self.device)
            L_avg=F.smooth_l1_loss(q_avg,TD_target.detach())

            #Critic Loss-function
            target1=(reward+self.gamma*q1(next_state,self.mu_target(next_state)*done_mask)).to(self.device)
            target2 = (reward + self.gamma * q2(next_state, self.mu_target(next_state) * done_mask)).to(
                self.device)


            #Critic Network Updating
            q1_loss=0.7*L_avg+0.3*F.smooth_l1_loss(q1(state,action),target1.detach())+0.001*F.smooth_l1_loss(q1(state,action),q_avg.detach())
            q2_loss=0.7*L_avg+0.3*F.smooth_l1_loss(q2(state,action),target2.detach())+0.001*F.smooth_l1_loss(q2(state,action),q_avg.detach())



            q1_optim.zero_grad()
            q2_optim.zero_grad()
            q1_loss.backward(retain_graph=True)
            q2_loss.backward()
            q1_optim.step()
            q2_optim.step()




            q_avg=q1(state,action)
            q_avg+=q2(state,action)

            q_avg/=2.0
            #Actor Loss-function
            # Actor Network updating
            mu_loss=-q_avg.mean()
            self.mu_optimizer.zero_grad()
            mu_loss.backward()
            self.mu_optimizer.step()


            self.soft_update(q1,q1_target)
            self.soft_update(q2, q2_target)
