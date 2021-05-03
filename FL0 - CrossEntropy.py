import gym
from gym.envs.registration import register
from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70


class Net(nn.Module):
    """
    This is our neural net that predicts an action given a state observation.
    """
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


def iterate_batches(env, net, batch_size):
    """
    Generate batches of episodes
    :param env: gym environment
    :param net: the neural net
    :param batch_size: the size of the batch
    :return: None; instead, it yields a batch
    """
    batch = []  # batch to yield
    episode_reward = 0.0  # initialize
    episode_steps = []  # list to contain the steps of an episode
    obs = env.reset()  # first observation
    sm = nn.Softmax(dim=1)  # we'll need to softmax the output of the network to compute probabilities.
    while True:     # we'll keep going until an the calling function has achieved performance,
        # and we don't need to generate a batch anymore
        obs_v = torch.FloatTensor([obs])  # convert observation to vector
        act_probs_v = sm(net(obs_v))        # run observation vector through softmax to get probabilities.
        act_probs = act_probs_v.data.numpy()[0]  # these are the action probabilities
        action = np.random.choice(len(act_probs), p=act_probs)  # sample an action based on probs
        next_obs, reward, is_done, _ = env.step(action)  # step
        episode_reward += reward   # accumulate reward
        step = EpisodeStep(observation=obs, action=action)  # convert info to EpisodeStep
        episode_steps.append(step)  # log the EpisodeStep
        if is_done: # did we end the episode?
            e = Episode(reward=episode_reward, steps=episode_steps) # log the episode
            batch.append(e) # add the episode to the batch
            episode_reward = 0.0  # reset episode reward to 0
            episode_steps = []  # reset episode steps array
            next_obs = env.reset() # reset environment
            if len(batch) == batch_size: # do we have enough episode in the batch?
                yield batch  # if so, yield it...
                batch = []   # reset it.
        obs = next_obs  # save the initial observation.


def filter_batch(batch, percentile):
    """
     Remove episodes from the batch that weren't good enough.
    :param batch: the batch in question...
    :param percentile: the cutoff
    :return:
    """
    rewards = list(map(lambda s: s.reward, batch))  # convert batch to a list of rewards
    reward_bound = np.percentile(rewards, percentile)  # computes the percentile-th of rewards as threshold
    reward_mean = float(np.mean(rewards)) # average rewards of the batch

    train_obs = []
    train_act = []
    # go through the batch of episodes, if the reward of that episode is GREATER than the threshold, include it in
    # training.
    for reward, steps in batch:
        if reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, steps))
        train_act.extend(map(lambda step: step.action, steps))

    train_obs_v = torch.FloatTensor(train_obs)  # convert to torch tensor
    train_act_v = torch.LongTensor(train_act)   # convert to torch tensor
    return train_obs_v, train_act_v, reward_bound, reward_mean


if __name__ == "__main__":
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

    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)

    for iter_no, batch in enumerate(iterate_batches(
            env, net, BATCH_SIZE)):
        obs_v, acts_v, reward_b, reward_m = \
            filter_batch(batch, PERCENTILE)
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()
        print("%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (
            iter_no, loss_v.item(), reward_m, reward_b))
        if reward_m > 199:
            print("Solved!")
            break
    writer.close()

