# Episodic multi DDPG

from episodic_multi_ddpg import episodic_multi_ddpg
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    agent1=episodic_multi_ddpg(150)
    agent2=episodic_multi_ddpg(150)
    agent3=episodic_multi_ddpg(150)
    agent4=episodic_multi_ddpg(150)
    agent5=episodic_multi_ddpg(150)
    agent6 = episodic_multi_ddpg(150)
    agent7 = episodic_multi_ddpg(150)
    agent8 = episodic_multi_ddpg(150)


    step1,out1=agent1.train(tau=0.1)
    step2,out2=agent1.train(tau=0.15)
    step3,out3=agent1.train(tau=0.2)
    step4,out4=agent1.train(tau=0.25)
    step5,out5=agent1.train(tau=0.3)
    step6,out6=agent1.train(tau=0.35)
    step7,out7=agent1.train(tau=0.4)
    step8,out8=agent1.train(tau=0.45)

    plt.title('EPisodic multi Agent tau')
    plt.plot(step1,out1)
    plt.plot(step2,out2)
    plt.plot(step3,out3)
    plt.plot(step4,out4)
    plt.plot(step5,out5)
    plt.plot(step6,out6)
    plt.plot(step7, out7)
    plt.plot(step8, out8)
    plt.legend(['tau=0.1','tau=0.15','tau=0.2','tau=0.25','tau=0.3','tau=0.35','tau=0.4','tau=0.45'])
    plt.xlabel('Number of steps')
    plt.ylabel('Cumulative reward')
    plt.show()