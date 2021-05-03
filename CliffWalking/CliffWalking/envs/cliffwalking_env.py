import gym
from gym import error, spaces, utils
from gym.utils import seeding
from copy import deepcopy

class CliffWalkingEnv(gym.Env):
    """
  This implements the Cliff Walking Environment as described in Sutton & Barto's RL book in Example 6.6.
  Essentially, it is a 4x12 grid. The start block is located in the first/bottom row, leftmost square. And the goal
  is at the first/bottom row, rightmost square. The whole first/bottom row between the start and goal is a cliff.
  Each move costs -1. Stepping into the cliff pays a reward of -1000, and sends the agent back to the start.
  The episode ends only when the goal is reached.
  """

    def __init__(self):
        self.rows = 4
        self.cols = 12
        self.start = [0, 0]
        self.goal = [0, 11]
        self.current_state = None

        # There are four actions: up, down, left and right
        self.action_space = spaces.Discrete(4)

        # observation is the x, y coordinate of the grid
        self.observation_space = spaces.Discrete(self.rows*self.cols)

    def reset(self):
        self.current_state = self.start
        return self.observation(self.current_state)

    def step(self, action):
        reward = -1.0
        new_state = deepcopy(self.current_state)

        if action == 0:  # left
            new_state[1] = max(new_state[1] - 1, 0)
        elif action == 1:  # down
            new_state[0] = max(new_state[0] - 1, 0)
        elif action == 2:  # right
            new_state[1] = min(new_state[1] + 1, self.cols - 1)
        elif action == 3:  # up
            new_state[0] = min(new_state[0] + 1, self.rows - 1)
        else:
            raise Exception("Invalid action.")
        self.current_state = new_state

        is_terminal = False
        if self.current_state[0] == 0 and self.current_state[1] > 0:
            if self.current_state[1] < self.cols - 1:
                reward = -100.0
                self.current_state = deepcopy(self.start)
            else:
                is_terminal = True

        return self.observation(self.current_state), reward, is_terminal, {}

    def render(self, mode='human'):
        pass

    def observation(self, state):
        return state[0] * self.cols + state[1]
