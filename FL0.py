from gym.envs.registration import register
import matplotlib.pyplot as plt
from helpers import q_learn, sarsa


register(
        id='FrozenLakeNotSlippery-v0',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name': '4x4', 'is_slippery': False},
        #    max_episode_steps=100,
        #    reward_threshold=0.78, # optimum = .8196
    )


def main():
    env = 'FrozenLakeNotSlippery-v0'
    num_episodes = 5000
    gamma = 0.1
    alpha = 0.05
    epsilon = 0.05
    Q_ave_rewards = q_learn(env, num_episodes, gamma, alpha, epsilon)
    SARSA_ave_rewards = sarsa(env, num_episodes, gamma, alpha, epsilon)

    fig, ax = plt.subplots()
    ax.plot(Q_ave_rewards, 'b', label='QLearning')
    ax.plot(SARSA_ave_rewards, 'r', label='SARSA')

    ax.legend()
    ax.set(xlabel='num episodes', ylabel='ave reward',
           title=' Frozen Lake Static')
    ax.grid()

    plt.show()


if __name__ == "__main__":
    main()

