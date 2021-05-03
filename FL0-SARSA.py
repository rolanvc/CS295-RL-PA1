import collections
import gym
from gym.envs.registration import register
from helpers import sarsa
import matplotlib.pyplot as plt





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
    ave_rewards = sarsa(env, num_episodes, gamma, alpha, epsilon)

    fig, ax = plt.subplots()
    ax.plot(ave_rewards, 'b', label='average rewards')
    # ax.plot(nonzero_states, 'g', label='nonzero states')
    ax.legend()
    ax.set(xlabel='num episodes', ylabel='ave reward',
           title=' Non-Slip Frozen lake, SARSA, Ave Reward per episode vs. Num of Episodes')
    ax.grid()

    plt.show()
    filename = "plots/{}-{}-gamma={}-alpha={}-epsilon={}.png".format("Static Frozen Lake", "SARSA" ,gamma, alpha, epsilon)
    fig.savefig(filename)


if __name__ == "__main__":
    main()
