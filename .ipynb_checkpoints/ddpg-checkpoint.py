# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 17:01:06 2022

@author: duzen
"""

import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from models import *
from utils import *

#torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPGagent:
    def __init__(self, env, hidden_size=256, actor_learning_rate=1e-4, critic_learning_rate=1e-3, gamma=0.99, tau=0.01, max_memory_size=1000000):
        # Params
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.gamma = gamma
        self.tau = tau
        self.actor_losses = []
        self.critic_losses = []

        # Networks
        self.actor = Actor(self.num_states, hidden_size, self.num_actions).to(device)
        self.actor_target = Actor(self.num_states, hidden_size, self.num_actions).to(device)
        self.critic = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions).to(device)
        self.critic_target = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions).to(device)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        # Training
        self.memory = Memory(max_memory_size)        
        #self.critic_criterion  = nn.SmoothL1Loss()
        self.critic_criterion  = nn.MSELoss()
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate,weight_decay = 1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
        
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for par in self.actor_target.parameters():
            par.requires_grad = False
        for par in self.critic_target.parameters():
            par.requires_grad = False
        
        
            
    def get_action(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0)).to(device)
        action = self.actor.forward(state)
        action = action.detach().cpu().numpy()[0]
        return action
    
    def update(self, batch_size):
        states, actions, rewards, next_states, _ = self.memory.sample(batch_size)
       # print(rewards)
        
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.FloatTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        
        # Critic update 
        self.critic_optimizer.zero_grad()
        Qvals = self.critic.forward(states, actions)
        with torch.no_grad():
            next_actions = self.actor_target.forward(next_states)
            next_Q = self.critic_target.forward(next_states, next_actions.detach())
            Qprime = rewards + self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime)
        
        self.critic_losses.append(critic_loss.item())
        
        critic_loss.backward() 
        self.critic_optimizer.step()
        
        for p in self.critic.parameters():
            p.requires_grad = False
        
        # Actor update
        self.actor_optimizer.zero_grad()
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()
        
        self.actor_losses.append(policy_loss.item())
        
        policy_loss.backward()
        self.actor_optimizer.step()
        
        for p in self.critic.parameters():
            p.requires_grad = True

        with torch.no_grad():
        # update target networks 
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
           
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))