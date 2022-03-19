import gym
from gym import spaces

import numpy as np
import pandas as pd
import torch

torch.manual_seed(0)

class Decode_v1(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, factors_returns=None, strategy_returns=None, window=5, random_start=True):
        super(Decode_v1, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(-1, 1, (11,))
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(-1, 1, (11 * 2 + 2,))

        self.factors_returns = factors_returns
        self.strategy_returns = strategy_returns

        # computation of deviation
        self.window = window
        self.deviation = factors_returns.rolling(window).std()
        self.deviation /= self.deviation.max()
        self.deviation = self.deviation.fillna(0)

        self.random_start = random_start
        self.last_index = len(factors_returns)
        self.first_index = 0 if not random_start else np.random.randint(window, self.last_index - 6)
        self.current_index = self.first_index

        self.weights_list = []
        self.weights_df = None

        self.max_factor_value = factors_returns.max().max()

    def _get_observation(self):
        factors = self.factors_returns.values[self.current_index]
        pred_return = (self.weights_list[self.current_index] * self.factors_returns.values[self.current_index]).sum()
        strategy_return = self.strategy_returns.values[self.current_index]
        deviation = self.deviation.values[self.current_index]
        obs = np.concatenate((factors, deviation,strategy_return,[pred_return]))
        return obs

    def get_reward(self, weights):
        if self.current_index < self.window :
            weights = weights[self.first_index:self.current_index]
            pred_returns = (1 + (weights * self.factors_returns[self.first_index:self.current_index]).sum(axis=1)).cumprod().pct_change().fillna(0)
            tracking_error = (pred_returns.values - self.strategy_returns[self.first_index:self.current_index].iloc[:,0].values)
        else :
            weights = weights[self.current_index - self.window:self.current_index]
            pred_returns = (1 + (weights* self.factors_returns[self.current_index - self.window :self.current_index]).sum(
                axis=1)).cumprod().pct_change().fillna(0)
            tracking_error = (pred_returns.values - self.strategy_returns[self.current_index - self.window:self.current_index].iloc[:,0].values)
        
        turn_over = (weights - weights.shift(1)).abs().fillna(0).values / self.window
        error_terms = np.concatenate([tracking_error, turn_over.flatten()], axis=0)
            
        return -np.sqrt(np.mean(error_terms ** 2)) + (weights.values[-1].min() -  weights.values[-1].max())/100           

    def step(self, action):
        done = self.current_index == self.last_index - 1
        if not done:
            self.current_index += 1
            if torch.is_tensor(action):
                action = action.cpu().numpy()
            self.weights_list = np.concatenate((self.weights_list,action.reshape(1,-1)))
            weights = pd.DataFrame(index=self.factors_returns[self.first_index:self.current_index+1].index,
                                   columns=self.factors_returns[self.first_index:self.current_index+1].columns)
            weights[:] = self.weights_list
            self.weights_df = weights
            info = {}
            reward = self.get_reward(weights)
            observation = self._get_observation()
            return observation, reward, done, info
        else:
            info = {}
            reward = self.get_reward(self.weights_df)
            observation = self._get_observation()
            return observation, reward, done, info

    def reset(self):
        self.first_index = 0 if not self.random_start else np.random.randint(self.window, self.last_index - 6)
        self.current_index = self.first_index
        self.weights_list = np.zeros((1,11))
        self.weights_df = pd.DataFrame(self.weights_list,index=self.factors_returns[0:1].index, columns = self.factors_returns[0:1].columns)
        observation = self._get_observation()
        return observation  

    def render(self, mode='human'):
        state = self._get_observation()
        return state

    def close(self):
        pass
