import copy
from typing import Tuple
from util import OrnsteinUhlenbeckProcess,AdaptiveParamNoiseSpec,ddpg_distance_metric
from ReplayBuffer import ReplayBuffer
from ddpg_vanilla import ddpg_vanilla
import numpy as np
import gym
import tqdm
# Environment Hyper parameter
env_name = 'Pendulum-v1'
env1=gym.make(env_name)
env2=gym.make(env_name)
env3=gym.make(env_name)

class vanilla_ddpg:

    def __init__(self,num_episode:int,batch_size:int=64):
        self.num_episode=num_episode
        self.batch_size=batch_size

        self.ddpg1=ddpg_vanilla(num_episode)


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
        step_list_ddpg=0
        step_count=0
        total_rewards=0

        ddpg1=self.ddpg1

        param_noise=self.param_noise

        for frame_idx in tqdm.trange(self.num_episode):

            s1=env1.reset()
            total_reward=0
            step_counter=0
            noise_counter=0
            ddpg1.perturb_actor_parameters(self.param_noise)


            done1=0


            while True:

                if not done1:
                    ddpg1.noise.reset()
                    a1=ddpg1.action(s1,ddpg1.noise.step(),param_noise)
                    s1_1,r1,done1,_=env1.step(a1)


                    transition=(s1,a1,r1/100.,s1_1,done1)
                    self.memory.add(transition)

                    total_reward+=r1
                    step_counter+=1
                    noise_counter+=1
                    s1=s1_1

                    if len(self.memory)>=self.batch_size:
                        ddpg1.train(self.memory,batch_size=self.batch_size)




                if done1 :

                    break


            running_reward_ddpg.append(total_reward)
            step_count=step_count+step_counter
            step_list_ddpg.append(step_count)

        step_list_ddpg=np.array(step_list_ddpg)
        out=self.numpy_ewma_vectorized_v2(np.array(running_reward_ddpg),20)
        return step_list_ddpg,out