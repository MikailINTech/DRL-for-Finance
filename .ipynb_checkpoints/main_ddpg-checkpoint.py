# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 17:01:58 2022

@author: duzen
"""

import sys
import numpy as np
import gym
import pandas as pd
import matplotlib.pyplot as plt
from itertools import count
from datetime import datetime
from env import Decode_v1
from ddpg import DDPGagent
from utils import *

dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d')
factors_returns = pd.read_csv('factors_returns.csv', index_col=0, 
                      parse_dates=True, date_parser=dateparse)
strategy_returns = pd.read_csv('strategy_returns.csv', index_col=0, 
                      parse_dates=True, date_parser=dateparse)

env = Decode_v1(factors_returns=factors_returns,strategy_returns=strategy_returns)

#env = gym.make('MountainCarContinuous-v0')

agent = DDPGagent(env)
noise = OUNoise(env.action_space)
batch_size = 64
rewards = []
avg_rewards = []

for episode in range(50):
    state = env.reset()
    noise.reset()
    episode_reward = 0
    
    for step in count():
        action = agent.get_action(state)
        action = noise.get_action(action, step)
        new_state, reward, done, _ = env.step(action) 
        agent.memory.push(state, action, reward, new_state, done)
        
        if len(agent.memory) > batch_size:
            agent.update(batch_size)        
        
        state = new_state 
        episode_reward += reward

        if done:
            sys.stdout.write("episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))
            break

    rewards.append(episode_reward)
    avg_rewards.append(np.mean(rewards[-10:]))

plt.plot(rewards)
plt.plot(avg_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()