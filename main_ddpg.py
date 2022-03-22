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
batch_size = 128
nb_episodes = 200
rewards = []
avg_rewards = []

best_env = env
best_reward = -100

for episode in range(nb_episodes):
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
            print(f'Env reward is {freward(env.weights_df)}')
            if freward(env.weights_df) > best_reward :
                best_env = copy.deepcopy(env)
                best_reward = freward(env.weights_df)
                print(f'best reward has been set to {best_reward}')
                print('\x1b[6;30;42m' + 'best_env has been changed' + '\x1b[0m')
            break

    rewards.append(episode_reward)
    avg_rewards.append(np.mean(rewards))

plt.plot(rewards,label='Rewards')
plt.plot(avg_rewards,label='Average rewards')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.show()