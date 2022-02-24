import numpy as np
import os
from typing import Dict,List,Tuple


class ReplayBuffer:

    def __init__(self,obs_dim:int,size:int,batch_size:int):

        self.obs_buf=np.zeros([size,obs_dim],dtype=np.float32)
        self.next_obs_buf=np.zeros([size,obs_dim],dtype=np.float32)
        self.acts_buf=np.zeros([size],dtype=np.float32)
        self.rews_buf=np.zeros([size],dtype=np.float32)
        self.done_buf=np.zeros([size],dtype=np.float32)


        self.max_size,self.batch_size=size,batch_size
        self.ptr,self.size=0,0


    def store(self,obs:np.ndarray,act:np.ndarray,rew:float,next_obs:np.ndarray,done:bool):

        done=0. if done else 1.

        self.obs_buf[self.ptr]=obs
        self.next_obs_buf[self.ptr]=next_obs
        self.acts_buf[self.ptr]=act
        self.rews_buf[self.ptr]=rew
        self.done_buf[self.ptr]=done

        self.ptr=(self.ptr+1)%self.max_size
        self.size=min(self.size+1,self.max_size)


    def sample_batch(self)->Dict[str,np.ndarray]:

        idxs=np.random.choice(self.size,self.batch_size,replace=False)

        obs=self.obs_buf[idxs]
        acts=self.acts_buf[idxs]
        next_obs=self.next_obs_buf[idxs]
        rews=self.rews_buf[idxs]
        done=self.done_buf[idxs]

        return dict(obs=obs,next_obs=next_obs,acts=acts,rews=rews,done=done)


    def __len__(self):
        return self.size