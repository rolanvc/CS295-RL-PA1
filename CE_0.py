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
import matplotlib.pyplot as plt

import random

# In[20]:


import gym
import gym.spaces
random.seed(123456)
env = gym.make('FrozenLake-v0', is_slippery=False)


# In[21]:


class OneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(OneHotWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (env.observation_space.n, ), dtype=np.float32)

    def observation(self, observation):
        r = np.copy(self.observation_space.low)
        r[observation] = 1.0
        return r

env = OneHotWrapper(env)


# ## The Agent
#  ### The Model

# In[22]:


obs_size = env.observation_space.shape[0] # 16
n_actions = env.action_space.n  # 4
HIDDEN_SIZE = 32


net= nn.Sequential(
            nn.Linear(obs_size, HIDDEN_SIZE),
            nn.Sigmoid(),
            nn.Linear(HIDDEN_SIZE, n_actions)
        )


# ### Get an Action

# In[23]:


sm = nn.Softmax(dim=1)

def select_action(state):
        state_t = torch.FloatTensor([state])
        act_probs_t = sm(net(state_t))
        act_probs = act_probs_t.data.numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs)
        return action


# ### Optimizer and Loss function

# In[24]:


import torch.optim as optim

objective = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=net.parameters(), lr=0.001)


# ## Training the Agent

# In[25]:


BATCH_SIZE = 100

GAMMA = 0.9

PERCENTILE = 30
REWARD_GOAL = 0.8

from collections import namedtuple

Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


# In[26]:


def graph_rewards(rewards_mean):
    f, sp = plt.subplots()
    sp.plot(rewards_mean, 'b', label='Average rewards of batch')
    sp.legend()
    sp.set(xlabel='num episodes', ylabel='ave reward',
           title='Average Batch Rewards')
    
    sp.grid()
    plt.show()


# In[27]:


iter_no = 0
reward_mean = 0
full_batch = []
batch = []
episode_steps = []
episode_reward = 0.0
state = env.reset()
mean_rewards = []

while reward_mean < REWARD_GOAL:
        action = select_action(state)
        next_state, reward, episode_is_done, _ = env.step(action)

        episode_steps.append(EpisodeStep(observation=state, action=action))
        episode_reward += reward
        
        if episode_is_done: # Episode finished            
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            next_state = env.reset()
            episode_steps = []
            episode_reward = 0.0
             
            if len(batch) == BATCH_SIZE: # New set of batches ready --> select "elite"
                reward_mean = float(np.mean(list(map(lambda s: s.reward, batch))))
                elite_candidates= batch 
                returnG = list(map(lambda s: s.reward * (GAMMA ** len(s.steps)), elite_candidates))
                reward_bound = np.percentile(returnG, PERCENTILE)

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


# In[29]:


graph_rewards(mean_rewards)


# ## Test the Agent

# In[15]:


test_env = OneHotWrapper(gym.make('FrozenLake-v0', is_slippery=False))
state= test_env.reset()
test_env.render()

is_done = False

while not is_done:
    action = select_action(state)
    new_state, reward, is_done, _ = test_env.step(action)
    test_env.render()
    state = new_state

print("reward = ", reward)


# ----
# 
# DEEP REINFORCEMENT LEARNING EXPLAINED - 07
# # **Cross-Entropy Method Performance Analysis**
# ## Implementation of the Cross-Entropy Training Loop

# ### Base line

# In[ ]:




