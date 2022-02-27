import copy
from typing import Tuple
from util import OrnsteinUhlenbeckProcess,AdaptiveParamNoiseSpec,ddpg_distance_metric
from ReplayBuffer import ReplayBuffer
from ddpg_vanilla_multi import ddpg_vanilla_multi
import numpy as np
import gym
import tqdm
# Environment Hyper parameter
env_name = 'Pendulum-v1'
env1=gym.make(env_name)
env2=gym.make(env_name)
env3=gym.make(env_name)

class vanilla_multi_ddpg:

    def __init__(self,num_episode:int,batch_size:int=64):
        self.num_episode=num_episode
        self.batch_size=batch_size

        self.ddpg1=ddpg_vanilla_multi(num_episode)
        self.ddpg2=ddpg_vanilla_multi(num_episode)
        self.ddpg3=ddpg_vanilla_multi(num_episode)

        self.memory=ReplayBuffer()

        self.param_noise=AdaptiveParamNoiseSpec(initial_stddev=0.05,desired_action_stddev=0.3,adaptation_coefficient=1.05)


    def numpy_ewma_vectorized_v2(self,data, window):

        alpha = 2 / (window + 1.0)
        alpha_rev = 1 - alpha
        n = data.shape[0]

        pows = alpha_rev ** (np.arange(n + 1))

        scale_arr = 1 / pows[:-1]
        offset = data[0] * pows[1:]
        pw0 = alpha * alpha_rev ** (n - 1)

        mult = data * pw0 * scale_arr
        cumsums = mult.cumsum()
        out = offset + cumsums * scale_arr[::-1]
        return out

    def train(self)->Tuple:

        running_reward_ddpg=list()
        step_list_ddpg=list()
        step_count=0
        total_rewards=list()
        R_max=-1000000
        ddpg1=self.ddpg1
        ddpg2=self.ddpg2
        ddpg3=self.ddpg3
        param_noise=self.param_noise

        for frame_idx in tqdm.trange(self.num_episode):

            s1=env1.reset()
            s2=env2.reset()
            s3=env3.reset()
            total_reward=[0,0,0]
            step_counter=[0,0,0]

            noise_counter=0
            ddpg1.perturb_actor_parameters(self.param_noise)
            ddpg2.perturb_actor_parameters(self.param_noise)
            ddpg3.perturb_actor_parameters(self.param_noise)

            done1,done2,done3=0,0,0


            while True:

                if not done1:
                    ddpg1.noise.reset()
                    a1=ddpg1.action(s1,ddpg1.noise.step(),param_noise)
                    s1_1,r1,done1,_=env1.step(a1)


                    transition=(s1,a1,r1/100.,s1_1,done1)
                    self.memory.add(transition)

                    total_reward[0]+=r1
                    step_counter[0]+=1
                    noise_counter+=1
                    s1=s1_1

                    if len(self.memory)>=self.batch_size:
                        ddpg1.train(self.memory,batch_size=self.batch_size)


                if not done2:
                    ddpg2.noise.reset()
                    a2=ddpg2.action(s2,ddpg2.noise.step(),param_noise)
                    s1_2,r2,done2,_=env2.step(a2)


                    transition=(s2,a2,r2/100.,s1_2,done2)
                    self.memory.add(transition)


                    total_reward[1]+=r2
                    step_counter[1]+=1
                    noise_counter+=1
                    s2=s1_2

                    if len(self.memory)>=self.batch_size:
                        ddpg2.train(self.memory,batch_size=self.batch_size)


                if not done3:
                    ddpg3.noise.reset()
                    a3=ddpg3.action(s3,ddpg3.noise.step(),param_noise)
                    s1_3,r3,done3,_=env3.step(a3)


                    transition=(s3,a3,r3/100.,s1_3,done3)
                    self.memory.add(transition)


                    total_reward[2]+=r3
                    step_counter[2]+=1
                    noise_counter+=1
                    s3=s1_3

                    if len(self.memory)>=self.batch_size:
                        ddpg3.train(self.memory,batch_size=self.batch_size)


                if done1 and done2 and done3:

                    break


            ddpgs=[ddpg1,ddpg2,ddpg3]
            total_rewards.append(total_reward)
            idx=np.array(total_reward).argmax()

            myddpg=ddpgs[idx]

            ddpg1=copy.deepcopy(myddpg)
            ddpg2=copy.deepcopy(myddpg)
            ddpg3=copy.deepcopy(myddpg)


            if self.memory.position-noise_counter>0:
                noise_data=self.memory.data[self.memory.position-noise_counter:self.memory.position]
            else:
                noise_data=self.memory.data[self.memory.position-noise_counter+60000:60000]\
                +self.memory.data[0:self.memory.position]

            noise_data=np.array(noise_data)
            noise_s,noise_a,_,_,_=zip(*noise_data)

            perturbed_actions=noise_a
            unperturbed_actions=myddpg.action(np.array(noise_s),None,None)
            ddpg_dist=ddpg_distance_metric(perturbed_actions,unperturbed_actions)
            param_noise.adapt(ddpg_dist)

            running_reward_ddpg.append(total_reward[idx])
            step_count=step_count+step_counter[idx]
            step_list_ddpg.append(step_count)

        step_list_ddpg=np.array(step_list_ddpg)
        out=self.numpy_ewma_vectorized_v2(np.array(running_reward_ddpg),20)
        return step_list_ddpg,out