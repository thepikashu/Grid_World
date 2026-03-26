import random 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class DQN(nn.Module):
  def __init__(self, n_observations, n_actions, hidden_nn_size=128): 
    super(DQN, self).__init__()
    self.layer1 = nn.Linear(n_observations, hidden_nn_size)
    self.layer2 = nn.Linear(hidden_nn_size,hidden_nn_size)
    self.layer3 = nn.Linear(hidden_nn_size, n_actions)

  def forward(self, x):
    x = F.relu(self.layer1(x))
    x = F.relu(self.layer2(x))
    return self.layer3(x)

class Memory(object):
  def __init__(self, capacity):
    self.memory = deque([], maxlen=capacity)

  def push(self, *args): 
    self.memory.append(Transition(*args))

  def sample(self, batch_size):
    return random.sample(self.memory, batch_size)

  def __len__(self):
    return len(self.memory)

class Action(): 
    def __init__(self, action_space_len, nn):
        self.n = action_space_len
        self.nn = nn
    
    # Updated to accept 'epsilon' as a keyword argument from train.py
    def select(self, state, epsilon=0):
        sample = random.random()
        
        # If random sample is higher than epsilon, use the Neural Network (Exploit)
        if sample > epsilon:
            with torch.no_grad():
                # self.nn(state) gives Q-values for all actions; .max(1)[1] picks the index of the highest
                a = self.nn(state).max(1)[1].view(1, 1)
                # print('[NN action]', a ) # Optional: uncomment for debugging
                return a
        # Otherwise, take a random action (Explore)
        else:
            a = torch.tensor([[random.randint(0, self.n - 1)]])
            # print('[Random Action]', a) # Optional: uncomment for debugging
            return a
        
    
