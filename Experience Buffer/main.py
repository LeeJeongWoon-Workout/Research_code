from Episodic_DDPG import DDPGAgent_epi
from Vanilla_DDPG import DDPGAgent_van
from Episodic_MO_DDPG import DDPGAgent_epi_mo
from Episodic_M_DDPG import DDPGAgent_epi_m
from My_Version_DDPG import DDPGAgent
import gym
import matplotlib.pyplot as plt
import numpy as np
import tqdm

if __name__ == "__main__":
    # environment
    env_id = "Pendulum-v1"
    env = gym.make(env_id)

    #P=0.2
    #Sim=0.5
    #HyperParameter
    #Memory_size=5e4
    #Epoch=10
    num_frames=5000
    memory_size=50000
    batch_size=64
    epoch=10
    result_van=np.zeros([int(num_frames/5)],dtype=np.float32)
    result_epi=np.zeros([int(num_frames/5)],dtype=np.float32)
    result_epi_mo=np.zeros([int(num_frames/5)],dtype=np.float32)
    result_epi_m=np.zeros([int(num_frames/5)],dtype=np.float32)
    result_my=np.zeros([int(num_frames/5)],dtype=np.float32)


    for i in tqdm.trange(epoch):
        # DDPGAgent
        agent5= DDPGAgent(env,memory_size,batch_size)
        agent1 = DDPGAgent_van(env, memory_size, batch_size)
        agent2 = DDPGAgent_epi(env, memory_size, batch_size)
        agent3 = DDPGAgent_epi_mo(env, memory_size, batch_size)
        agent4 = DDPGAgent_epi_m(env, memory_size, batch_size)

        # Reward_list

        result5=agent5.train(num_frames)
        result1 = agent1.train(num_frames)
        result2 = agent2.train(num_frames)
        result3 = agent3.train(num_frames)
        result4 = agent4.train(num_frames)

        #Accumulate the result

        result_my += result5
        result_van += result1
        result_epi+=result2
        result_epi_mo+=result3
        result_epi_m+=result4


    #Visualization
    plt.title('Experience Buffer')
    plt.xlabel('episode')
    plt.ylabel('Averaged Reward (10 epochs)')
    plt.plot(result_van/epoch)
    plt.plot(result_epi/epoch)
    plt.plot(result_epi_mo/epoch)
    plt.plot(result_epi_m/epoch)
    plt.plot(result_my/epoch)
    plt.legend(['Vanilla','Episodic','MO','M','My'])
    plt.show()