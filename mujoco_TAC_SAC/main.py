import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
from replay_memory import ReplayMemory
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="Humanoid-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--title' ,default="Humanoid-v2")
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.001, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--hmemory_tau1',type=float,default=0.0,metavar='G')
parser.add_argument('--hmemory_tau2',type=float,default=0.05,metavar='G')
parser.add_argument('--hmemory_tau3',type=float,default=0.1,metavar='G')
parser.add_argument('--hmemory_tau4',type=float,default=0.15,metavar='G')
parser.add_argument('--hmemory_tau5',type=float,default=0.2,metavar='G')
parser.add_argument('--epoch',type=int,default=1000,metavar='G')
parser.add_argument('--render',type=bool,default=False,metavar='G')
parser.add_argument('--algo',default='sac',metavar='G')
args = parser.parse_args()

# Environment
# env = NormalizedActions(gym.make(args.env_name))

env = gym.make(args.env_name)
env.seed(args.seed)
env.action_space.seed(args.seed)
result_list=list()
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)
agent2=SAC(env.observation_space.shape[0], env.action_space, args)
agent3=SAC(env.observation_space.shape[0], env.action_space, args)
agent4=SAC(env.observation_space.shape[0], env.action_space, args)
agent5=SAC(env.observation_space.shape[0], env.action_space, args)
print(env.action_space.shape[0])
#Tesnorboard

# Memory
memory = ReplayMemory(args.replay_size, args.seed)
hmemory=ReplayMemory(args.replay_size,args.seed)


memory2 = ReplayMemory(args.replay_size, args.seed)
hmemory2=ReplayMemory(args.replay_size,args.seed)


memory3 = ReplayMemory(args.replay_size, args.seed)
hmemory3=ReplayMemory(args.replay_size,args.seed)


memory4 = ReplayMemory(args.replay_size, args.seed)
hmemory4=ReplayMemory(args.replay_size,args.seed)


memory5 = ReplayMemory(args.replay_size, args.seed)
hmemory5=ReplayMemory(args.replay_size,args.seed)


num_list1=list()
reward_list1=list()

num_list2=list()
reward_list2=list()


num_list3=list()
reward_list3=list()

num_list4=list()
reward_list4=list()

num_list5=list()
reward_list5=list()

# Training Loop
total_numsteps1= 0
total_numsteps2 = 0
total_numsteps3 = 0
total_numsteps4 = 0
total_numsteps5 = 0
updates = 0
R_MAX=-1000000000000000
R_MAX2=-100000000000000
R_MAX3=-100000000000000
R_MAX4=-100000000000000
R_MAX5=-100000000000000
for i_episode in range(args.epoch):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()
    transition_list=list()
    while not done:
        if args.render:
            env.render()
        if args.start_steps > total_numsteps1:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > args.batch_size and len(hmemory)>args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(args.hmemory_tau1,hmemory,memory, args.batch_size, updates)


                updates += 1

        next_state, reward, done, _ = env.step(action) # Step
        episode_steps += 1
        total_numsteps1 += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        memory.push(state, action, reward, next_state, mask) # Append transition to memory
        transition_list.append([state,action,reward,next_state,mask])

        state = next_state

        if episode_reward>R_MAX:
            R_MAX=episode_reward
            for t in transition_list:
                hmemory.push(*t)

    if total_numsteps1 > args.num_steps:
        break

    print("1 Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps1, episode_steps, round(episode_reward, 2)))
    num_list1.append(total_numsteps1)
    reward_list1.append(round(episode_reward,2))

for i_episode in range(args.epoch):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()
    transition_list=list()
    while not done:
        if args.render:
            env.render()
        if args.start_steps > total_numsteps2:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent2.select_action(state)  # Sample action from policy

        if len(memory2) > args.batch_size and len(hmemory2)>args.batch_size :
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent2.update_parameters(args.hmemory_tau2,hmemory2,memory2, args.batch_size, updates)


                updates += 1

        next_state, reward, done, _ = env.step(action) # Step
        episode_steps += 1
        total_numsteps2 += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        memory2.push(state, action, reward, next_state, mask) # Append transition to memory
        transition_list.append([state,action,reward,next_state,mask])

        state = next_state

        if episode_reward>R_MAX2:
            R_MAX2=episode_reward
            for t in transition_list:
                hmemory2.push(*t)

    if total_numsteps2 > args.num_steps:
        break

    print("2 Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps2, episode_steps, round(episode_reward, 2)))
    num_list2.append(total_numsteps2)
    reward_list2.append(round(episode_reward,2))

for i_episode in range(args.epoch):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()
    transition_list=list()
    while not done:
        if args.render:
            env.render()
        if args.start_steps > total_numsteps3:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent3.select_action(state)  # Sample action from policy

        if len(memory3) > args.batch_size and len(hmemory3)>args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent3.update_parameters(args.hmemory_tau3,hmemory3,memory3, args.batch_size, updates)


                updates += 1

        next_state, reward, done, _ = env.step(action) # Step
        episode_steps += 1
        total_numsteps3 += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        memory3.push(state, action, reward, next_state, mask) # Append transition to memory
        transition_list.append([state,action,reward,next_state,mask])

        state = next_state

        if episode_reward>R_MAX3:
            R_MAX3=episode_reward
            for t in transition_list:
                hmemory3.push(*t)

    if total_numsteps3 > args.num_steps:
        break

    print("3 Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps3, episode_steps, round(episode_reward, 2)))
    num_list3.append(total_numsteps3)
    reward_list3.append(round(episode_reward,2))

for i_episode in range(args.epoch):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()
    transition_list=list()
    while not done:
        if args.render:
            env.render()
        if args.start_steps > total_numsteps4:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent4.select_action(state)  # Sample action from policy

        if len(memory4) > args.batch_size and len(hmemory4)>args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent4.update_parameters(args.hmemory_tau4,hmemory4,memory4, args.batch_size, updates)


                updates += 1

        next_state, reward, done, _ = env.step(action) # Step
        episode_steps += 1
        total_numsteps4 += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        memory4.push(state, action, reward, next_state, mask) # Append transition to memory
        transition_list.append([state,action,reward,next_state,mask])

        state = next_state

        if episode_reward>R_MAX4:
            R_MAX4=episode_reward
            for t in transition_list:
                hmemory4.push(*t)

    if total_numsteps4 > args.num_steps:
        break

    print("4 Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps4, episode_steps, round(episode_reward, 2)))
    num_list4.append(total_numsteps4)
    reward_list4.append(round(episode_reward,2))

for i_episode in range(args.epoch):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()
    transition_list=list()
    while not done:
        if args.render:
            env.render()
        if args.start_steps > total_numsteps5:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent5.select_action(state)  # Sample action from policy

        if len(memory5) > args.batch_size and len(hmemory5)>args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent5.update_parameters(args.hmemory_tau5,hmemory5,memory5, args.batch_size, updates)


                updates += 1

        next_state, reward, done, _ = env.step(action) # Step
        episode_steps += 1
        total_numsteps5 += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        memory5.push(state, action, reward, next_state, mask) # Append transition to memory
        transition_list.append([state,action,reward,next_state,mask])

        state = next_state

        if episode_reward>R_MAX5:
            R_MAX5=episode_reward
            for t in transition_list:
                hmemory5.push(*t)

    if total_numsteps5 > args.num_steps:
        break

    print("5 Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps5, episode_steps, round(episode_reward, 2)))
    num_list5.append(total_numsteps5)
    reward_list5.append(round(episode_reward,2))


env.close()

plt.title('{},action_space: {},state_space: {}'.format(args.title,env.action_space.shape[0],env.observation_space.shape[0]))
plt.plot(num_list1,reward_list1)
plt.plot(num_list2,reward_list2)
plt.plot(num_list3,reward_list3)
plt.plot(num_list4,reward_list4)
plt.plot(num_list5,reward_list5)
plt.legend(['tau=0','tau=0.05','tau=0.1','tau=0.15','tau=0.20'])
plt.xlabel('environment interactions')
plt.ylabel('Accumulated Return')
plt.show()