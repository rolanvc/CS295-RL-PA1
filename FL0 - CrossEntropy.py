import gym
from gym.envs.registration import register
from collections import namedtuple
import numpy as np
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 90


class Net(nn.Module ):
    """
    This is our neural net that predicts an action given a state observation.
    """
    def __init__(self, obs_size, hidden_size, n_actions, device):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        ).to(device)

    def forward(self, x):
        return self.net(x)


Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        print("Will use: {}". format(torch.cuda.get_device_name(0)))
        return torch.device('cuda')
    else:
        print("Will use: CPU")
        return torch.device('cpu')

num_batch = 0
has_elite = False

def generate_batches(device, env, net, batch_size, has_elite):
    """
    Generate batches of episodes.
    We continuously run episodes of the environment and add this to the  batch until we have a total of batch_size AND
    the total rewards in the batch > 1.0 (to avoid falling into the zero hole)
    :param env: gym environment
    :param net: the neural net
    :param batch_size: the size of the batch
    :return: None; instead, it yields a batch/array of episodes.
    """
    batch = []  # batch to yield
    total_batch_reward = 0.0  # we'll track total batch reward. If 0.0, we keep generating episodes.
    episode_reward = 0.0  # initialize
    episode_steps = []  # list to contain the steps of an episode
    obs = env.reset()  # first observation
    sm = nn.Softmax(dim=0)  # we'll need to softmax the output of the network to compute probabilities.
    while True:     # we'll keep going until an the calling function has achieved performance,
        # and we don't need to generate a batch anymore
        obs_v = torch.FloatTensor([obs])  # convert observation to vector
        obs_v = obs_v.to(device)
        act_probs_v = sm(net(obs_v))        # run observation vector through softmax to get probabilities.
        act_probs = act_probs_v.detach().cpu().numpy()  # these are the action probabilities
        if  has_elite:  # check if we've previously had en elite batch of data
            # if so, we use the probabilities of the output of tne network
            action = np.random.choice(len(act_probs), p=act_probs)
        else:
            # otherwise, we sample as if from a uniform distribution.
            action = np.random.choice(len(act_probs))
        next_obs, reward, is_done, _ = env.step(action)  # step
        episode_reward += reward   # accumulate episode reward
        total_batch_reward += reward # accumulate total_batch_reward
        step = EpisodeStep(observation=obs, action=action)  # convert info to EpisodeStep
        episode_steps.append(step)  # log the EpisodeStep
        if is_done: # did we end the episode?
            e = Episode(reward=episode_reward, steps=episode_steps) # log the episode
            batch.append(e) # add the episode to the batch
            episode_reward = 0.0  # reset episode reward to 0
            episode_steps = []  # reset episode steps array
            next_obs = env.reset()  # reset environment
            if len(batch) >= batch_size and (total_batch_reward > 0.0 or len(batch) > 100):  # do we have enough episode in the batch?
                yield batch  # if so, yield it...
                batch = []   # reset it.
        obs = next_obs  # save the initial observation.


def filter_for_elite(batch, percentile, device):
    """
     Filter batch. We only want to train with the top percentile.
    :param batch: the batch in question...
    :param percentile: the cutoff
    :return:
    """
    global has_elite
    rewards = list(map(lambda s: s.reward, batch))  # convert batch to a list of rewards
    reward_bound = np.percentile(rewards, percentile)  # computes the percentile-th of rewards as threshold
    reward_m =float(np.mean(rewards))
    elite_mean_rewards = 0

    train_obs = []
    train_act = []
    # go through the batch of episodes, if the reward of that episode is GREATER than the threshold, include it in
    # training.
    total = 0
    for reward, steps in batch:
        if reward >= reward_bound:
            elite_mean_rewards += reward
            total += 1
            train_obs.extend(map(lambda step: step.observation, steps))
            train_act.extend(map(lambda step: step.action, steps))

    if total > 0:
        has_elite = True
    elite_mean_rewards /= total
    train_obs_v = torch.FloatTensor(train_obs)  # convert to torch tensor
    train_obs_v_new = train_obs_v.reshape(-1, 1).to(device)
    train_act_v = torch.LongTensor(train_act).to(device)   # convert to torch tensor
    return train_obs_v_new, train_act_v, reward_bound, reward_m, elite_mean_rewards


def graph(rewards, elite_rewards):
    fig, ax = plt.subplots()
    ax.plot(rewards, 'b', label='Average rewards of batch')
    ax.plot(elite_rewards, 'g', label='Average Elite Rewards')
    ax.legend()
    ax.set(xlabel='num episodes', ylabel='ave reward',
           title=' Ave Reward per episode vs. Num of Episodes')
    ax.grid()

    plt.show()
    filename = "plots/cross-entropy-frozen-lake.png"
    fig.savefig(filename)


def main():
    register(
        id='FrozenLakeNotSlippery-v0',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name': '4x4', 'is_slippery': False},
    )
    env = gym.make('FrozenLakeNotSlippery-v0')
    # env = gym.make("CartPole-v0")  # Define environment.
    # env = gym.wrappers.Monitor(env, directory="mon", force=True)
    obs_size = 1
    n_actions = 4
    device = get_default_device()
    reward_arr = []
    elite_reward_arr = []

    net = Net(obs_size, HIDDEN_SIZE, n_actions, device)
    net = net.to(device)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)

    print("Start at:{}".format(datetime.datetime.now()))
    for iter_no, batch in enumerate(generate_batches(
            device, env, net, BATCH_SIZE, has_elite)):
        obs_v, acts_v, reward_b, reward_m, elite_reward = \
            filter_for_elite(batch, PERCENTILE, device)
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()
        elite_reward_arr.append(elite_reward)
        reward_arr.append(reward_m)
        print("%d: loss=%.3f, reward_mean=%.2f, elite_mean= %.2f rw_bound=%.2f" % (
            iter_no, loss_v.item(), reward_m, elite_reward, reward_b))
        if reward_m > .95 or iter_no > 2000:
            print("Solved! at {}".format(datetime.datetime.now()))
            break
    graph(reward_arr, elite_reward_arr)


if __name__ == "__main__":
    main()
