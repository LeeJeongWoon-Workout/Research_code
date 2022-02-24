import gym
import matplotlib.pyplot as plt
import numpy as np
import tqdm


from SAMC7_C2 import SAMC7_C2
from SAMC7_C4 import SAMC7_C4
from SAMC8_C4 import SAMC8_C4
from SAMC10_C4 import SAMC10_C4

if __name__ == "__main__":
    # environment
    env_id = "Pendulum-v1"
    env = gym.make(env_id)


    num_frames=2000
    memory_size=50000
    batch_size=64
    epoch=1
    result_samc72=np.zeros([int(num_frames/10)],dtype=np.float32)
    result_samc74 = np.zeros([int(num_frames / 10)], dtype=np.float32)
    result_samc84 = np.zeros([int(num_frames / 10)], dtype=np.float32)
    result_samc104 = np.zeros([int(num_frames / 10)], dtype=np.float32)


    for i in tqdm.trange(epoch):
        agent1=SAMC7_C2(env,memory_size,batch_size,tau=0.25)
        agent2 = SAMC7_C4(env, memory_size, batch_size, tau=0.25)
        agent3 = SAMC8_C4(env, memory_size, batch_size, tau=0.25)
        agent4 = SAMC10_C4(env, memory_size, batch_size, tau=0.25)


        # Reward_list
        result1 = agent1.train(num_frames,a=0.35,b=0.65,w=0.01)
        result2 = agent1.train(num_frames, a=0.35, b=0.65, w=0.01)
        result3 = agent1.train(num_frames, a=0.35, b=0.65, w=0.01)
        result4 = agent1.train(num_frames, a=0.35, b=0.65, w=0.01)
        #result1 = agent1.train(num_frames)




        result_samc72+=result1
        result_samc74+=result2
        result_samc84+=result3
        result_samc104+=result4

    #Visualization
    plt.title('Ensemble main 1')
    plt.xlabel('episode (X10)')
    plt.ylabel('Averaged Reward')
    plt.plot(result_samc72)
    plt.plot(result_samc74)
    plt.plot(result_samc84)
    plt.plot(result_samc104)
    plt.legend(['SAMC7_C2','SAMC7_C4','SAMC8_C4','SAMC10_C4'])
    plt.show()