import torch
import gym
import collections
import numpy
from gym.envs.registration import register

# #############################
# #
# #  Some checking of CUDA
# #
# ############################3
cuda_ok = torch.cuda.is_available()
print("CUDA is available:", cuda_ok)
if (cuda_ok):
    print("CUDA DEVICE IS:", torch.cuda.get_device_name(0))




# ##################
#
# Define the function to choose an action
#
# #################


def get_action(o, step):
    actions = [1, 1, 2, 2, 1, 2]
    return actions[step]


# ##################
#
#  Register and define the Non-Slippery Frozen Lake
#
# #################
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)
env = gym.make('FrozenLakeNotSlippery-v0')

# #################
# parameters
NUM_EPISODES = 1
gamma = 1
alpha = 1
epsilon = 0.05

for i_episode in range(NUM_EPISODES):
    state = env.reset() # Initialize S
    farthest = state
    farthest_state = collections.defaultdict(float)
    rewards = collections.defaultdict(float)

    done = False
    step = 0
    print("New State is:{}".format(state))
    while not done:  # Loop until end of episode
        env.render()
        action = get_action(state, step)  # Choose A from S using policy (e.g. epsilon-greedy)
        newstate, reward, done, info = env.step(action)  # Take action A, observe R, S'
        # qvalues[(state, action)] = qvalues[(state, action)] + alpha*(reward + gamma * get_max_Q(newstate) - qvalues[(state, action)])
        state = newstate
        print("New State is:{}".format(newstate))
        if newstate > farthest:
            farthest = newstate

        rewards[i_episode] += reward
        step +=1

    print("For episode:{}, total rewards is: {}".format(i_episode, rewards[i_episode]))
    print("For episode:{}, farthest state is: {}".format(i_episode, farthest))


