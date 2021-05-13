#!/usr/bin/env python
# coding: utf-8

# DEEP REINFORCEMENT LEARNING EXPLAINED - 06
# # **Solving Frozen-Lake Environment With Cross-Entropy Method**
# ## Agent Creation Using Deep Neural Networks

#  
# 
# ## The Environment

# In[19]:


import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import random

# In[20]:


import gym
import gym.spaces
random.seed(123456)


# In[21]:


class OneHotWrapper(gym.ObservationWrapper):
    """
    This is a wrapper class for the environment. So we could create wrapper functions around it.
    """
    def __init__(self, env):
        super(OneHotWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (env.observation_space.n, ), dtype=np.float32)

    def observation(self, observation):
        r = np.copy(self.observation_space.low)
        r[observation] = 1.0
        return r


class Net(nn.Module):
    """
    This is the neural network to predict policy given state.
    """
    def __init__(self, obs_size, hidden_size, n_actions):
        """
        This network has only 1 hidden layer.
        :param obs_size: the size of the observation
        :param hidden_size: the hidden layer size.
        :param n_actions: the number of actions to predict.
        """
        super(Net, self).__init__()
        self.net= nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        """
        Forwarding an input into the layer.
        :param x:
        :return:
        """
        return self.net(x)


def select_action(net, state):
    """
    This selects an action from a given neural network, and input state.
    :param net: The neural network that computes the policy.
    :param state: The input state
    :return: the action
    """
    sm = nn.Softmax(dim=1)
    state_t = torch.FloatTensor([state])
    y = net.forward(state_t)
    act_probs_t = sm(y)
    act_probs = act_probs_t.data.numpy()[0]
    action = np.random.choice(len(act_probs), p=act_probs)
    return action


# ### Optimizer and Loss function

# In[24]:





# ## Training the Agent

# In[25]:


#BATCH_SIZE = 100

#GAMMA = 0.9

#PERCENTILE = 30
#REWARD_GOAL = 0.8

from collections import namedtuple

Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


def cross_entropy(net, env, gamma, batch_size, percentile, reward_goal):
    """
    This is the main cross entropy algorithm.
    :param net: The neural network
    :param env: the environment.
    :param gamma: the discount factor
    :param batch_size: batch_size
    :param percentile: percentile cut-off to be considered elite
    :param reward_goal: the reward goal that signals termination of algorithm.
    :return: average rewards
    """
    iter_no = 0
    reward_mean = 0
    full_batch = []
    batch = []
    episode_steps = []
    episode_reward = 0.0
    state = env.reset()
    mean_rewards = []

    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.001)
    while reward_mean < reward_goal:
        action = select_action(net, state)
        next_state, reward, episode_is_done, _ = env.step(action)

        episode_steps.append(EpisodeStep(observation=state, action=action))
        episode_reward += reward
        
        if episode_is_done: # Episode finished            
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            next_state = env.reset()
            episode_steps = []
            episode_reward = 0.0
             
            if len(batch) == batch_size: # New set of batches ready --> select "elite"
                reward_mean = float(np.mean(list(map(lambda s: s.reward, batch))))
                elite_candidates= batch 
                returnG = list(map(lambda s: s.reward * (gamma ** len(s.steps)), elite_candidates))
                reward_bound = np.percentile(returnG, percentile)

                train_obs = []
                train_act = []
                elite_batch = []
                for example, discounted_reward in zip(elite_candidates, returnG):
                        if discounted_reward > reward_bound:
                              train_obs.extend(map(lambda step: step.observation, example.steps))
                              train_act.extend(map(lambda step: step.action, example.steps))
                              elite_batch.append(example)
                full_batch=elite_batch
                state=train_obs
                acts=train_act

                
                if len(full_batch) != 0 : # just in case empty during an iteration
                 state_t = torch.FloatTensor(state)
                 acts_t = torch.LongTensor(acts)
                 optimizer.zero_grad()
                 action_scores_t = net(state_t)
                 loss_t = objective(action_scores_t, acts_t)
                 loss_t.backward()
                 optimizer.step()
                 mean_rewards.append(reward_mean)
                 print("%d: loss=%.3f, reward_mean=%.3f \r" % (iter_no, loss_t.item(), reward_mean))
                 iter_no += 1
                batch = []
        state = next_state

    return mean_rewards


def main():
    env = gym.make('FrozenLake-v0', is_slippery=False)
    env = OneHotWrapper(env)

    obs_size = env.observation_space.shape[0]  # 16
    n_actions = env.action_space.n  # 4
    batch_size= 100
    percentile = 70
    reward_goal = 0.70
    plt.rcParams['figure.figsize'] = [40, 20]
    f, (sp1, sp2) = plt.subplots(1,2)

    gamma0 = 0.9
    net0 = Net(obs_size=obs_size, hidden_size=32, n_actions=n_actions)
    mean_rewards0 = cross_entropy(net0, env, gamma0, batch_size, percentile, reward_goal)
    episodes = [e for e in range(0, len(mean_rewards0)*batch_size) if e % batch_size== 0]
    sp1.plot(episodes,mean_rewards0, 'b', label='Average rewards at gamma={}'.format(gamma0))

    gamma1 = 0.5
    net1 = Net(obs_size=obs_size, hidden_size=64, n_actions=n_actions)
    mean_rewards1 = cross_entropy(net1, env, gamma1, batch_size, percentile, reward_goal)
    episodes = [e for e in range(0, len(mean_rewards1)*batch_size) if e % batch_size== 0]
    sp1.plot(episodes, mean_rewards1, 'r', label='Average rewards at gamma={}'.format(gamma1))

    gamma2 = 0.2
    net2 = Net(obs_size=obs_size, hidden_size=64, n_actions=n_actions)
    mean_rewards2 = cross_entropy(net2,env, gamma2, batch_size, percentile, reward_goal)
    episodes = [e for e in range(0, len(mean_rewards2)*batch_size) if e % batch_size== 0]
    sp1.plot(episodes, mean_rewards2, 'g', label='Average rewards at gamma={}'.format(gamma2))

    sp1.legend()
    sp1.set(xlabel='num episodes', ylabel='ave reward',
            title='Average Rewards, varying Discount Rate')
    sp1.grid()

    net3 = Net(obs_size=obs_size, hidden_size=32, n_actions=n_actions)
    gamma3 = 0.9
    mean_rewards3 = cross_entropy(net3, env, gamma3, batch_size, percentile, reward_goal)
    episodes = [e for e in range(0, len(mean_rewards3) * batch_size) if e % batch_size == 0]
    sp2.plot(episodes, mean_rewards3, 'b', label='Ave R at gamma={}, hidden size=32'.format(gamma0))

    net4 = Net(obs_size=obs_size, hidden_size=64, n_actions=n_actions)
    gamma1 = 0.5
    mean_rewards4 = cross_entropy(net4, env, gamma1, batch_size, percentile, reward_goal)
    episodes = [e for e in range(0, len(mean_rewards4) * batch_size) if e % batch_size == 0]
    sp2.plot(episodes, mean_rewards4, 'r', label='Ave R at gamma={}, hidden size=64'.format(gamma1))

    net5 = Net(obs_size=obs_size, hidden_size=128, n_actions=n_actions)
    gamma5 = 0.2
    mean_rewards5 = cross_entropy(net5, env, gamma5, batch_size, percentile, reward_goal)
    episodes = [e for e in range(0, len(mean_rewards5) * batch_size) if e % batch_size == 0]
    sp2.plot(episodes, mean_rewards5, 'g', label='Ave R at gamma={}, hidden size=128'.format(gamma2))

    sp2.legend()
    sp2.set(xlabel='num episodes', ylabel='ave reward',
           title='Average Rewards, varying Discount Rate and Hidden Size')
    sp2.grid()
    plt.show()


if __name__ == "__main__":
    main()



# ## Test the Agent

# In[15]:


testing_env = OneHotWrapper(gym.make('FrozenLake-v0', is_slippery=False))
testing_state = testing_env.reset()
testing_env.render()

is_done = False

while not is_done:
    action = select_action(testing_state)
    new_state, reward, is_done, _ = testing_env.step(action)
    testing_env.render()
    testing_state = new_state

print("reward = ", reward)


# ----
# 
# DEEP REINFORCEMENT LEARNING EXPLAINED - 07
# # **Cross-Entropy Method Performance Analysis**
# ## Implementation of the Cross-Entropy Training Loop

# ### Base line

# In[ ]:




