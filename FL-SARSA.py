import torch
import gym
import collections
import numpy
import operator
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
NUM_EPISODES = 10000
gamma = 1
alpha = 1
epsilon = 0.10


# Support Functions
def get_action( state ):
    """ Select action from state state using epsilon-greedy:
        Parameters: state: s,

        Returns: action:int
    """
    # Extract the action values.
    action_values = [qvalues[(state, 0)], qvalues[(state, 1)],qvalues[(state, 2)], qvalues[(state, 3)]]

    # Then initialize their probabilities as uniform over epsilon.
    action_probs = [epsilon/4, epsilon/4, epsilon/4, epsilon/4]

    # Find the max action value's index, then add (1-epsilon) to its probability.
    maxQsa = max(action_values)

    # if the maximum is 0.0, then all values are still 0.0, therefore force a uniform distribution..
    if maxQsa == 0.0:
        action_probs = [0.25, 0.25, 0.25, 0.25]
    else:
        action = action_values.index(maxQsa)
        action_probs[action] = action_probs[action] + 1-epsilon

    # sample over the probabilities
    return numpy.random.choice([0, 1, 2, 3], p=action_probs )


def get_max_Q(state):
    """ Given a state, return the max Qvalue among all the actions

    :param state:
    :return: max Qvalue among all the actions
    """
    action_values = [qvalues[(state, 0)], qvalues[(state, 1)],qvalues[(state, 2)], qvalues[(state, 3)]]
    maxQsa = max(action_values)
    return maxQsa


def printAction(action):
    return ["Left", "Down", "Right", "Up"][action]


# Q-Values
Q_values = collections.defaultdict(float)
farthest_state = {}
actions = {}
rewards = {}
for i_episode in range(NUM_EPISODES):  # repeat for each episode
    state = env.reset()  # Initialize S
    farthest = state
    actions[i_episode] = []  # log the actions for this episode
    rewards[i_episode] = 0.0  # log the rewards fo this episode
    done = False
    while not done:  # Loop until end of episode
        action = get_action(state)  # Choose A from S using policy (e.g. epsilon-greedy)
        actions[i_episode].append(action)
        newstate, reward, done, info = env.step(action)  # Take action A, observe R, S'
        Q_values[(state, action)] = Q_values[(state, action)] + alpha*(reward + gamma * get_max_Q(newstate) - Q_values[(state, action)])
        state = newstate
        if newstate > farthest:
            farthest = newstate
        rewards[i_episode] += reward
    farthest_state[i_episode] = farthest
    if i_episode % 10000 == 0:
        print("Finished episode:{} of {}".format(i_episode, NUM_EPISODES))

best_episode = max(farthest_state.items(), key=operator.itemgetter(1))[0]
print("Farthest was at episode {} at state={}".format(best_episode, farthest_state[best_episode]))
print("Steps were:{}".format(actions[best_episode]))
best_reward_ep = max(rewards.items(), key=operator.itemgetter(1))[0]
print("Best reward was {} at episode {}".format(rewards[best_reward_ep], best_reward_ep))





