from Episodic_DDPG import DDPGAgent_epi
from Vanilla_DDPG import DDPGAgent_van
import gym
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # environment
    env_id = "Pendulum-v1"
    env = gym.make(env_id)

    #HyperParameter
    num_frames=5000
    memory_size=10000
    batch_size=32

    #DDPG Agent Object
    agent1=DDPGAgent_epi(env,memory_size,batch_size)
    agent2=DDPGAgent_van(env,memory_size,batch_size)

    #Reward_list
    result_epi=agent1.train(num_frames)
    result_van=agent2.train(num_frames)

    #Visualization
    plt.title('Experience Buffer')
    plt.xlabel('episode (X50)')
    plt.ylabel('Averaged Reward')
    plt.plot(result_epi)
    plt.plot(result_van)
    plt.legend(['Episodic','Vanilla'])
    plt.show()
