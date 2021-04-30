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
NUM_EPISODES = 100
gamma = 1
alpha = 1
epsilon = 0.05

# Q-Values
qvalues = collections.defaultdict(float)

# Support Functions
def get_action( state ):
    """ Select action from state state using epsilon-greedy:
        Parameters: state: s,

        Returns: action:int
    """
    # Extract the action values.
    action_values = [qvalues[(state, 0)], qvalues[(state, 1)],[(state, 2)], qvalues[(state, 3)] ]

    # Then initialize their probabilities as uniform over epsilon.
    action_probs = [epsilon/4, epsilon/4, epsilon/4, epsilon/4]

    # Find the max action value's index, then add (1-epsilon) to its probability.
    maxQsa = max(action_values)
    action = action_values.index(maxQsa)
    action_probs[action] = action_probs[action] + 1-epsilon

    # sample over the probabilities
    return numpy.random.choice([0, 1, 2, 3], action_probs )




for i_episode in range(NUM_EPISODES):
    observation = env.reset()
    action = get_action(o)
    
env.close()


