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
        
        self.weights_list = []
        self.weights_df = None


    def get_reward(self, weights):
        if len(self.weights) == 0:
            return 0
        pred_returns = (1 + (weights * self.factors_returns[:self.current_index]).sum(axis=1)).cumprod(
          ).pct_change().fillna(0)
        tracking_error =  (pred_returns.values - self.strategy_returns[:self.current_index].iloc[:,0].values
          ) * np.sqrt(250) * np.sqrt(weights.shape[1]+1)
        turn_over = 0.0020 * 365 * ((weights - weights.shift(1)).abs().fillna(0).values
          ) / ((weights.index[-1] -weights.index[0]).days) * np.sqrt(
          weights.shape[0] * (weights.shape[1]+1)) 
        error_terms = np.concatenate([tracking_error, turn_over.flatten()], axis=0)
        return -np.sqrt(np.mean(error_terms**2))
  
    
    def step(self, action):
        done = self.current_index == self.last_index
        if not done :
            self.current_index += 1
        self.weights_list.append(action.cpu().numpy())
        weights = pd.DataFrame(index=self.factors_returns[:self.current_index].index, columns = self.factors_returns[:self.current_index].columns)
        weights[:] = self.weights_list
        self.weights_df = weights
        info = {}
        reward = self.get_reward(weights)
        observation = self.factors_returns.values[self.current_index]
        
        return observation, reward, done, info



    def reset(self):
        self.current_index = 0
        self.weights = []
        observation = self.factors_returns.values[self.current_index]
        return observation  # reward, done, info can't be included
    
    def render(self, mode='human'):
        state = self.factors_returns.values[self.current_index]
        return state
    
    def close (self):
        pass