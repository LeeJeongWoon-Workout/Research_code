#2. EPisodic multi Vanilla multi Vanilla DDPG Episodic DDPG

from episodic_multi_ddpg import episodic_multi_ddpg
from vanilla_multi_ddpg import vanilla_multi_ddpg
from episodic_vanilla_ddpg import episodic_vanilla_ddpg
from vanilla_ddpg import vanilla_ddpg

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    agent1=episodic_multi_ddpg(20)
    agent2=vanilla_multi_ddpg(20)
    agent3=episodic_vanilla_ddpg(20)
    agent4=vanilla_ddpg(20)


    step1,out1=agent1.train(tau=0.25)
    step2,out2=agent2.train()
    step3,out3=agent3.train(tau=0.25)
    step4,out4=agent4.train()


    plt.title('EPisodic multi Agent tau')
    plt.plot(step1,out1)
    plt.plot(step2,out2)
    plt.plot(step3,out3)
    plt.plot(step4,out4)

    plt.legend(['episodic_multi_ddpg','vanilla_multi_ddpg','episodic_vanilla_ddpg','vanilla_ddpg'])
    plt.xlabel('Number of steps')
    plt.ylabel('Cumulative reward')
    plt.show()