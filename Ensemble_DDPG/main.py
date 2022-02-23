import gym
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from Ensemble_DDPG.Vanilla_Episodic_DDPG import DDPGAgent_epi

if __name__ == "__main__":
    # environment
    env_id = "Pendulum-v1"
    env = gym.make(env_id)


    num_frames=100
    memory_size=50000
    batch_size=64
    epoch=5
    result_vanilla=np.zeros([int(num_frames/10)],dtype=np.float32)



    for i in tqdm.trange(epoch):

        agent1=DDPGAgent_epi(env,memory_size,batch_size,tau=0.25)

        # Reward_list

        result1=agent1.train(num_frames)

        #Accumulate the result

        result_vanilla+=result1


    #Visualization
    plt.title('Experience Buffer')
    plt.xlabel('episode (X10)')
    plt.ylabel('Averaged Reward')
    plt.plot(result_vanilla/10)
    plt.legend(['Vanilla'])
    plt.show()