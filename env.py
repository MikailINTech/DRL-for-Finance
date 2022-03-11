import gym
from gym import spaces

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



class Decode_v1(gym.Env):


    metadata = {'render.modes': ['human']}

    def __init__(self,factors_returns=None,strategy_returns=None):
        super(Decode_v1, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(-1,1,(11,))
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(-1,1,(11,))
        
        self.factors_returns = factors_returns
        self.strategy_returns = strategy_returns

        self.current_index = 0
        self.last_index = len(factors_returns)

        self.weights = []


    def get_reward(self, weights):
      pred_returns = (1 + (weights * self.factors_returns).sum(axis=1)).cumprod(
          ).pct_change().fillna(0)
      tracking_error =  (pred_returns.values - self.strategy_returns.iloc[:,0].values
          ) * np.sqrt(250) * np.sqrt(weights.shape[1]+1)
      turn_over = 0.0020 * 365 * ((weights - weights.shift(1)).abs().fillna(0).values
          ) / ((weights.index[-1] -weights.index[0]).days) * np.sqrt(
          weights.shape[0] * (weights.shape[1]+1)) 
      error_terms = np.concatenate([tracking_error, turn_over.flatten()], axis=0)
      return -np.sqrt(np.mean(error_terms**2))
    
    def step(self, action):
        self.weights.append(action.reshape(-1))
        done = False
        info = {}
        if self.current_index == self.last_index-1:
          weights = pd.DataFrame(index=self.factors_returns.index, columns = self.factors_returns.columns, data=self.weights)
          reward = self.get_reward(weights)
          reward = 1/ (np.abs(reward)+0.0001)
          done = True
          observation = self.factors_returns.values[self.current_index]
          return observation, reward, done, info
        
        self.current_index += 1
        observation = self.factors_returns.values[self.current_index]
        if self.current_index == 1:
            reward = 0
            return observation, reward, done, info

        
        weights = pd.DataFrame(index=self.factors_returns.index[:self.current_index], columns = self.factors_returns.columns, data=self.weights)
        pred_returns = (1 + (weights[:self.current_index] * self.factors_returns[:self.current_index]).sum(axis=1)).cumprod().pct_change().fillna(0)
        #print(pred_returns[self.current_index-1])
        #print(self.strategy_returns)
        if pred_returns[self.current_index-1] >= 0.001:
            reward = 0
            return observation, reward, done, info

        reward = 0.01/ (np.abs((pred_returns[self.current_index-1] - self.strategy_returns.iloc[self.current_index].values[0]))+ 0.001)
        
        return observation, reward, done, info


    def reset(self):
        self.current_index = 0
        self.weights = []
        observation = self.factors_returns.values[self.current_index]
        return observation  # reward, done, info can't be included
    def render(self, mode='human'):
        pass
    def close (self):
        pass