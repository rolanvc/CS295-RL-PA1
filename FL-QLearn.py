import torch
import gym
import collections
import numpy
import matplotlib.pyplot as plt
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
alpha = 0.5
epsilon = 0.10

# Q-Values

# Support Functions
def get_action( state, q_values ):
    """
    Select an action from [0,1,2,3] (representing ["Left", "Down", "Right", "Up"] using epsilon-greedy.
    :param state: the state of interest
    :param q_values: a map of q_values.
    :return: an action
    """
    # Extract the action values.
    action_values = [q_values[(state, 0)], q_values[(state, 1)],q_values[(state, 2)], q_values[(state, 3)]]

    # Then initialize their probabilities as uniform over epsilon.
    action_probs = [epsilon/4, epsilon/4, epsilon/4, epsilon/4]

    # Find the max action value's index, then add (1-epsilon) to its probability.
    maxqsa = max(action_values)

    # if the maximum is 0.0, then all values are still 0.0, therefore force a uniform distribution..
    if maxqsa == 0.0:
        action_probs = [0.25, 0.25, 0.25, 0.25]
    else:
        action = action_values.index(maxqsa)
        action_probs[action] = action_probs[action] + 1-epsilon

    # sample over the probabilities
    return numpy.random.choice([0, 1, 2, 3], p=action_probs )


def get_max_Q(state, q_values):
    """ Given a state, return the max Qvalue among all the actions
    :param state: the state of interest
    :param q_values: a dict with keys as (s,a) and values as q_values.
    :return: max Qvalue among all the actions
    """
    action_values = [q_values[(state, 0)], q_values[(state, 1)], q_values[(state, 2)], q_values[(state, 3)]]
    maxQsa = max(action_values)
    return maxQsa


def printAction(action):
    return ["Left", "Down", "Right", "Up"][action]


def update_ave_rewards(ave_rewards, episode, rewards):
    """
    update ave_rewards array. This array contains a list of successive ave rewards vs num_episodes.
    parameter episode is the index of the ith episode. Therefore its entry will be the average of the (episode-1)th average
    muliplied by (episode-1) then adding rewards, then dividing by episode. Need to figure out zero-indexing.

    :param ave_rewards: a list as described above. The value of  the index is the num of (episodes+1), and the value contains
        the average of rewards across those episodes
    :param episode: the episode index: int
    :param rewards: the scalar reward
    :return: None
    """
    assert episode == len(ave_rewards), "Episode:{} should equal len(ave_rewards):{}".format(episode, len(ave_rewards))
    if episode == 0:
        ave_rewards.append(rewards)
    else:
        reward_sum = ave_rewards[episode-1] * (episode-1)
        reward_sum += rewards
        reward_ave = reward_sum / episode
        ave_rewards.append(reward_ave)


def update_nonzero_q_count( nonzero_states, q_values, episode):
    """
    Update nonzero_states array with the count of non-zero q_values
    :param nonzero_states: an array with the count of non-zero q_values, index is episode.
    :param q_values: the dict of current q_values.
    :param episode: the current episode.

    :return:None
    """
    assert episode == len(nonzero_states), "Episode:{} should equal len(nonzero_states):{}".format(episode, len(nonzero_states))
    count = 0
    for k in q_values.keys():
        if q_values[k] > 0:
            count +=1

    nonzero_states.append(count)

#####################
#
# Start here
#
####################

Q_values = collections.defaultdict(float)  # dict of (state,action):value
actions = {}  # dict of episode:[actions]
rewards = {}  # dict of episode: total rewards for tha episode
ave_rewards = []  #  array of  ave rewards across all episodes until episode index

nonzero_states = []  # array of non-zero states. index is episode.
for i_episode in range(NUM_EPISODES):
    state = env.reset() # Initialize S
    farthest = state
    actions[i_episode] = []
    rewards[i_episode] = 0.0
    done = False
    while not done:  # Loop until end of episode
        action = get_action(state, Q_values)  # Choose A from S using policy (e.g. epsilon-greedy)
        actions[i_episode].append(action)
        newstate, reward, done, info = env.step(action)  # Take action A, observe R, S'
        Q_values[(state, action)] = Q_values[(state, action)] + alpha*(reward + gamma * get_max_Q(newstate, Q_values) - Q_values[(state, action)])
        state = newstate
        rewards[i_episode] += reward

    update_nonzero_q_count(nonzero_states, Q_values, i_episode)
    update_ave_rewards(ave_rewards, i_episode, rewards[i_episode])
    if i_episode % 10000 == 0:
        print("Finished episode:{} of {}".format(i_episode, NUM_EPISODES))


fig, ax = plt.subplots()
ax.plot(ave_rewards, 'b')
ax.plot(nonzero_states, 'g')

ax.set(xlabel='num episodes', ylabel='ave reward',
       title=' Ave Reward per episode vs. Num of Episodes')
ax.grid()

fig.savefig("test.png")
plt.show()





