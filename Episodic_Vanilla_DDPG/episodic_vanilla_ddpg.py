import copy
from typing import Tuple
from util import OrnsteinUhlenbeckProcess,AdaptiveParamNoiseSpec,ddpg_distance_metric
from ReplayBuffer import ReplayBuffer
from ddpg_episodic_vanilla import ddpg_episodic_vanilla
import numpy as np
import gym
import tqdm
# Environment Hyper parameter
env_name = 'Pendulum-v1'
env1=gym.make(env_name)
env2=gym.make(env_name)
env3=gym.make(env_name)

class episodic_vanilla_ddpg:

    def __init__(self,num_episode:int,batch_size:int=64):
        self.num_episode=num_episode
        self.batch_size=batch_size

        self.ddpg1=ddpg_episodic_vanilla(num_episode)


        self.memory=ReplayBuffer()
        self.hmemory=ReplayBuffer()

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

    def train(self,tau)->Tuple:

        running_reward_ddpg=list()
        step_list_ddpg=list()
        step_count=0
        total_rewards=list()
        ddpg1=self.ddpg1
        param_noise=self.param_noise
        R_max=-10000000

        for frame_idx in tqdm.trange(self.num_episode):

            s1=env1.reset()

            ddpg1.perturb_actor_parameters(self.param_noise)


            done1=0
            total_reward=0
            score1=0
            step_counter=0
            episode1=list()

            while True:

                if not done1:
                    ddpg1.noise.reset()
                    a1=ddpg1.action(s1,ddpg1.noise.step(),param_noise)
                    s1_1,r1,done1,_=env1.step(a1)

                    score1+=r1
                    transition=(s1,a1,r1/100.,s1_1,done1)
                    self.memory.add(transition)
                    episode1.append(transition)

                    total_reward+=r1
                    step_counter+=1
                    s1=s1_1

                    if len(self.memory)>=self.batch_size and len(self.hmemory)>=self.batch_size:
                        ddpg1.train(self.memory,self.hmemory,batch_size=self.batch_size,tau=tau)





                if done1:
                    if score1>R_max:
                        R_max=score1
                        for transition in episode1:
                            self.hmemory.add(transition)


                    break



            running_reward_ddpg.append(total_reward)
            step_count=step_count+step_counter
            step_list_ddpg.append(step_count)

        step_list_ddpg=np.array(step_list_ddpg)
        out=self.numpy_ewma_vectorized_v2(np.array(running_reward_ddpg),20)
        return step_list_ddpg,out