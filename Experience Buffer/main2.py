from Episodic_DDPG_tau import DDPGAgent_epi
import gym
import matplotlib.pyplot as plt
import numpy as np
import tqdm


if __name__ == "__main__":
    # environment
    env_id = "Pendulum-v1"
    env = gym.make(env_id)


    num_frames=5000
    memory_size=50000
    batch_size=64
    epoch=10
    result_1=np.zeros([int(num_frames/10)],dtype=np.float32)
    result_2 = np.zeros([int(num_frames / 10)], dtype=np.float32)
    result_3 = np.zeros([int(num_frames / 10)], dtype=np.float32)
    result_4 = np.zeros([int(num_frames / 10)], dtype=np.float32)
    result_5 = np.zeros([int(num_frames / 10)], dtype=np.float32)


    for i in tqdm.trange(epoch):
        # DDPGAgent
        agent1=DDPGAgent_epi(env,memory_size,batch_size)
        agent2 = DDPGAgent_epi(env,memory_size,batch_size)
        agent3 = DDPGAgent_epi(env,memory_size,batch_size)
        agent4 = DDPGAgent_epi(env,memory_size,batch_size)
        agent5 = DDPGAgent_epi(env,memory_size,batch_size)

        # Reward_list



        result1 = agent1.train(num_frames, tau=0.25)
        result2 = agent2.train(num_frames, tau=0.15)
        result3 = agent3.train(num_frames, tau=0.40)
        result4 = agent4.train(num_frames, tau=0.45)
        result5 = agent5.train(num_frames, tau=0.50)


        #Accumulate the result

        result_1+=result1
        result_2 += result2
        result_3 += result3
        result_4 += result4
        result_5 += result5



    #Visualization
    plt.title('Experience Buffer (Episodic P)')
    plt.xlabel('episode')
    plt.ylabel('Averaged Reward (20 epochs)')
    plt.plot(result_1)
    plt.plot(result_2)
    plt.plot(result_3)
    plt.plot(result_4)
    plt.plot(result_5)

    plt.legend(['tau=0.25','tau=0.15','tau=0.40','tau=0.45','tau=0.5'])
    plt.show()