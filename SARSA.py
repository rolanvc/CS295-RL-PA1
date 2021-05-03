import collections
import gym
from helpers import get_action, get_max_Q, update_ave_rewards, update_nonzero_q_count
import matplotlib.pyplot as plt


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
        farthest = state
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

    fig, ax = plt.subplots()
    ax.plot(ave_rewards, 'b', label='average rewards')
    # ax.plot(nonzero_states, 'g', label='nonzero states')
    ax.legend()
    ax.set(xlabel='num episodes', ylabel='ave reward',
           title=' Ave Reward per episode vs. Num of Episodes')
    ax.grid()

    plt.show()
    filename = "episodes={}-gamma={}-alpha={}-epsilon={}.png".format(num_episodes, gamma, alpha, epsilon)
    fig.savefig(filename)

from gym.envs.registration import register

def main():
    register(
        id='FrozenLakeNotSlippery-v0',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name': '4x4', 'is_slippery': False},
        #    max_episode_steps=100,
        #    reward_threshold=0.78, # optimum = .8196
    )
    env = 'FrozenLakeNotSlippery-v0'
    num_episodes = 5000
    gamma = 0.1
    alpha = 0.05
    epsilon = 0.05
    q_learn(env, num_episodes, gamma, alpha, epsilon)


if __name__ == "__main__":
    main()
