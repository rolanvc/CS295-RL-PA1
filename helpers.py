import numpy
import collections
import gym

# Support Functions


def get_action( state, q_values, epsilon):
    """
    Select an action from [0,1,2,3] (representing ["Left", "Down", "Right", "Up"] using epsilon-greedy.
    :param state: the state of interest
    :param q_values: a map of q_values.
    :param epsilon: prob of not greedy choice.
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


def q_learn(env_name, num_episodes, gamma, alpha, epsilon):
    """

    :param env_name:
    :param num_episodes:
    :param gamma:
    :param alpha:
    :param epsilon:
    :return:
    """

    Q_values = collections.defaultdict(float)  # dict of (state,action):value
    actions = {}  # dict of episode:[actions]
    rewards = {}  # dict of episode: total rewards for tha episode
    ave_rewards = []  # array of  ave rewards across all episodes until episode index
    nonzero_states = []  # array of non-zero states. index is episode.

    env = gym.make(env_name)
    for i_episode in range(num_episodes):
        state = env.reset()  # Initialize S
        actions[i_episode] = []
        rewards[i_episode] = 0.0
        done = False
        while not done:  # Loop until end of episode
            action = get_action(state, Q_values, epsilon)  # Choose A from S using policy (e.g. epsilon-greedy)
            actions[i_episode].append(action)
            newstate, reward, done, info = env.step(action)  # Take action A, observe R, S'
            Q_values[(state, action)] = Q_values[(state, action)] + alpha * (
                    reward + gamma * get_max_Q(newstate, Q_values) - Q_values[(state, action)])
            state = newstate
            rewards[i_episode] += reward

        update_nonzero_q_count(nonzero_states, Q_values, i_episode)
        update_ave_rewards(ave_rewards, i_episode, rewards[i_episode])
        if i_episode % 10000 == 0:
            print("Finished episode:{} of {}".format(i_episode, num_episodes))

    return ave_rewards


def sarsa(env_name, num_episodes, gamma, alpha, epsilon):
    """

    :param env_name:
    :param num_episodes:
    :param gamma:
    :param alpha:
    :param epsilon:
    :return:
    """

    actions = {}  # dict of episode:[actions]
    rewards = {}  # dict of episode: total rewards for tha episode
    ave_rewards = []  # array of  ave rewards across all episodes until episode index

    env = gym.make(env_name)

    q_values = collections.defaultdict(float)  # dict of (state,action):value
    for i_episode in range(num_episodes):

        state = env.reset()  # Initialize S
        action = get_action(state, q_values, epsilon)  # Choose A from S using policy (e.g. epsilon-greedy)
        actions[i_episode] = []
        actions[i_episode].append(action)
        rewards[i_episode] = 0.0
        done = False
        while not done:  # Repeat for each step of the episode
            newstate, reward, done, info = env.step(action)  # Take action A, observe R, S'
            newaction = get_action(newstate, q_values, epsilon)  # Choose A' from S' using policy (e.g. epsilon-greedy)
            q_values[(state, action)] = q_values[(state, action)] + alpha * (
                    reward + gamma * q_values[(newstate, newaction)] - q_values[(state, action)])
            state = newstate
            action = newaction
            rewards[i_episode] += reward

        update_ave_rewards(ave_rewards, i_episode, rewards[i_episode])
        if i_episode % 10000 == 0:
            print("Finished episode:{} of {}".format(i_episode, num_episodes))

    return ave_rewards
