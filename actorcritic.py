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
    



def normalize(data, mean, std, eps=1e-8):
    return (data-mean)/(std+eps)

def unnormalize(data, mean, std):
    return data*std+mean




class Actor(nn.Module):
    model_file="models/model"

    def __init__(self, obs_space, act_space):
        super().__init__()
        self.obs_dim = obs_space
        self.act_dim = act_space
        self.linear1 = nn.Linear(obs_space, 128)
        self.linear2 = nn.Linear(128, act_space)
        self.linear3 = nn.Linear(128, act_space)

        self.distribution = torch.distributions.Normal

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        mu = self.linear2(x)
        std = F.softplus(self.linear3(x)) + 0.001

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
        torch.save(self.state_dict(), self.model_file + "actor.pth" )

    def load(self, model_file):
        self.load_state_dict(torch.load(model_file + "actor.pth"))


class Critic(nn.Module):
    model_file="models/model"

    def __init__(self, obs_space, act_space):
        super().__init__()
        self.obs_dim = obs_space
        self.act_dim = act_space
        self.linear1 = nn.Linear(obs_space, 128)
        self.linear2 = nn.Linear(128,1)

        self.distribution = torch.distributions.Normal

    def forward(self, x):
        values = F.relu6(self.linear1(x))
        return self.linear2(values)

    def save(self):
        torch.save(self.state_dict(), self.model_file + "critic.pth" )

    def load(self, model_file):
        self.load_state_dict(torch.load(model_file + "critic.pth"))



class ActorCritic():
    def __init__(self, num_inputs, num_actions, learning_rate,
                 batch_size, policy_epochs, gamma = 0.99, entropy_coef=0.001):
        self.actor = Actor(num_inputs, num_actions)
        self.critic = Critic(num_inputs, num_actions)
        self.optimizerA = optim.Adam(self.actor.parameters(), lr=learning_rate[0])
        self.optimizerC = optim.Adam(self.actor.parameters(), lr=learning_rate[1])
        self.batch_size = batch_size
        self.policy_epochs = policy_epochs
        self.entropy_coef = entropy_coef
        self.gamma = gamma

        self.criticloss = nn.MSELoss()

    def update_actor(self, obs, actions, advantages):
        observations = torch.from_numpy(np.array(obs))
        actions = torch.from_numpy(np.array(actions))
        advantages = torch.from_numpy(advantages) 

        actions_distribution = self.actor.forward(observations.float())
        log_probs = actions_distribution.log_prob(actions)
        #assert log_probs.size() == advantages.size()
        loss = -(log_probs * advantages.view(-1,1)).sum()


        self.optimizerA.zero_grad()
        loss.backward()
        self.optimizerA.step()
        return(loss)

    def update_critic(self, obs, rewards):
        ob_no = torch.from_numpy(obs)
        reward_n = torch.from_numpy(rewards)

        v = (self.critic.forward(ob_no)).squeeze()
        v_next = torch.cat((v[1:],torch.tensor([0])))
        targets = reward_n + self.gamma * v_next.squeeze()

        loss = self.criticloss(v.float(), targets.float())
        
        self.optimizerC.zero_grad()
        loss.backward()
        self.optimizerC.step()
        targets.detach_()

        return loss

    def estimate_adv(self, obs, reward):
        current_obs = torch.from_numpy(obs)
        re_n = torch.from_numpy(reward)

        v = (self.critic.forward(current_obs)).squeeze()
        v_next = torch.cat((v[1:],torch.tensor([0])))
        Q = re_n + self.gamma * v_next.squeeze()

        adv_n = Q - v
        adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)

        return adv_n.detach().numpy()


    def discounted_return(self, rewards):

        n = len(rewards)
        list_of_discounted_cumsums= (self.gamma**(np.tile(np.arange(n),(n,1)))*rewards).sum(axis=1)

        return list_of_discounted_cumsums



    def train(self, env, seed=123):
        # SETTING SEED: it is good practice to set seeds when running experiments to keep results comparable
        np.random.seed(seed)
        torch.manual_seed(seed)
        env.seed(seed)

        loss = []
        epoch = self.policy_epochs
        # Training Loop
        for j in range(epoch):

            obs_l    = []
            action_l = []
            value_l  = []
            rewards_l = []

            ## Initialization
            done = False
            obs = env.reset()
            ## Collect rollouts
            while not done:
                if j%100 ==0 :
                    env.render()

                obs_l.append(obs)
                obs = torch.tensor(obs, dtype=torch.float32)
                action = self.actor.get_action(obs)
                value = self.critic(obs).detach().numpy()
                obs, reward, done, _ = env.step(action)
                #Noting the list of elements done
                action_l.append(action)
                rewards_l.append(reward)
                value_l.append(value)

            discount_r = self.discounted_return(np.array(rewards_l))
            obs_l = np.array(obs_l)
            l1 = self.update_critic(obs_l,discount_r)

            adv = self.estimate_adv(obs_l, discount_r)
            l2 = self.update_actor(obs_l,action_l,adv)

            loss.append([l1.item(),l2.item()])

            if j%100 ==0 or reward > 50 :
                print(f"Epoch {j} : Loss is {loss[-1]}")
                print(reward)
            
        return loss


    def test_policy(self, env):
        action_l = []
        obs = env.reset()
        done = False
        while not done:
            action = self.actor.get_action(obs)
            obs, reward, done, _ = env.step(action)
            action_l.append(action)

        return(np.array(action_l))

