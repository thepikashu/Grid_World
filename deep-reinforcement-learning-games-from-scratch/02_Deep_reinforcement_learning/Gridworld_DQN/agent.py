import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import namedtuple, deque

# Structure for memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class DQN(nn.Module):
    def __init__(self, n_obs, n_actions):
        super().__init__()
        self.f1 = nn.Linear(n_obs, 128)
        self.f2 = nn.Linear(128, 128)
        self.f3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        return self.f3(x)

class Memory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

# This is your DDQN "Sorcery" - Replace your train.py optimize function with this logic
def optimize_model(policy_net, target_net, memory, optimizer, BATCH_SIZE, GAMMA):
    if len(memory) < BATCH_SIZE:
        return
    
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    # Mask for non-terminal states
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    # Current Q-values
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # DDQN Logic: Select action with Policy Net, Evaluate with Target Net
    next_state_values = torch.zeros(BATCH_SIZE)
    with torch.no_grad():
        # Step 1: Policy net decides WHICH action is best
        best_actions = policy_net(non_final_next_states).argmax(1).unsqueeze(1)
        # Step 2: Target net tells us WHAT that action is worth
        next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, best_actions).squeeze(1)

    expected_q = (next_state_values * GAMMA) + reward_batch
    
    loss = nn.SmoothL1Loss()(state_action_values, expected_q.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
