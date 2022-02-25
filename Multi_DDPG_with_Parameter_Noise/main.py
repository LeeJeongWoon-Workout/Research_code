from multi_ddpg_parameter_noise import multi_parameter_noise_ddpg
import matplotlib.pyplot as plt
import numpy as plt

if __name__ == "__main__":

    agent=multi_parameter_noise_ddpg(200,batch_size=64)
    result=agent.train()

    plt.plot(result)
    plt.xlabel('episode')
    plt.ylabel('accumulative reward')
    plt.show()


