from turtle import forward
import gym
import numpy as np
from itertools import count
from typing import Any, cast
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from tqdm import tqdm

# Add softmax ?
# 


class PG_agent(nn.Module):
    model_file="pg_model.pth"

    def __init__(self, obs_space, act_space):
        super().__init__()
        self.linear1 = nn.Linear(obs_space, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.linear2 = nn.Linear(128, act_space)
        #self.linear3 = nn.Linear(128, act_space)
        log_std = -2* np.ones(act_space, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std))


    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.linear2(x)
        #mu = F.softmax(action_scores)
        mu = action_scores
        std = torch.exp(self.log_std)

        pi = Normal(mu,std)
        return pi

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # DONE: get this from HW1
            
        observation_tensor = torch.tensor(obs, dtype=torch.float)
        action_distribution = self.forward(observation_tensor)
        actions = action_distribution.sample()
        #actions = torch.clamp(actions, min=-1, max=1)
        # actions = F.softmax(action_distribution.sample()) In case of ?
        return cast(
            np.ndarray,
            actions.cpu().detach().numpy(),
        )

    def save(self):
        torch.save(self.state_dict(), self.model_file)

    def load(self, model_file):
        self.load_state_dict(torch.load(model_file))


class Policy():
    def __init__(self, num_inputs, num_actions, learning_rate,
                 batch_size, policy_epochs, entropy_coef=0.001):
        self.actor = PG_agent(num_inputs, num_actions)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.policy_epochs = policy_epochs
        self.entropy_coef = entropy_coef

    def update(self, obs, actions, rewards):
        observations = torch.from_numpy(np.array(obs))
        actions = torch.from_numpy(np.array(actions))
        advantages = torch.from_numpy(rewards) 
        actions_distribution = self.actor.forward(observations.float())
        log_probs = actions_distribution.log_prob(actions)
        #assert log_probs.size() == advantages.size()
        loss = -(log_probs * advantages.view(-1,1)).sum()


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return(loss)

    def discounted_return(self, rewards, gamma):

        n = len(rewards)
        list_of_discounted_cumsums= (gamma**(np.tile(np.arange(n),(n,1)))*rewards).sum(axis=1)

        return list_of_discounted_cumsums



    def train(self, env, seed=123, gamma=0.99):
        # SETTING SEED: it is good practice to set seeds when running experiments to keep results comparable
        np.random.seed(seed)
        torch.manual_seed(seed)
        env.seed(seed)

        loss = []
        epoch = self.policy_epochs
        # Training Loop
        for j in range(epoch):
            obs_l, action_l, rewards_l = [], [], []

            ## Initialization
            done = False
            obs = env.reset()
            ## Collect rollouts
            while not done:
                obs_l.append(obs)
                obs = torch.tensor(obs, dtype=torch.float32)
                action = self.actor.get_action(obs)
                obs, reward, done, _ = env.step(action)
                #Noting the list of elements done
                action_l.append(action)
                rewards_l.append(reward)

            discount_r = self.discounted_return(np.array(rewards_l),gamma)

            loss.append(self.update(obs_l, action_l, discount_r))

            if j%100 ==0 or reward > 50 :
                print(f"Epoch {j} : Loss is {loss[-1]}")
                print(reward)
    def test_policy(self, env):
        action_l = []
        obs = env.reset()
        done = False
        while not done:
            action = self.actor.get_action(obs)
            obs, reward, done, _ = env.step(action)
            action_l.append(action)

        return(np.array(action_l))
