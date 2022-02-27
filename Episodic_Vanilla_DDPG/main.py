from episodic_vanilla_ddpg import episodic_vanilla_ddpg
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    agent1=episodic_vanilla_ddpg(1000)
    step,out=agent1.train(tau=0.25)
    plt.plot(step,out)
    plt.title('Training reward over multiple runs')
    plt.xlabel('Number of steps')
    plt.ylabel('Cumulative reward')
    plt.show()
