import numpy as np

class GridWorld:
    def __init__(self, shape=(5, 5), obstacles=[], terminal=None):
        self.rows, self.cols = shape
        self.obstacles = obstacles
        self.terminal_state = terminal if terminal else (self.rows-1, self.cols-1)
        self.agent_pos = (0, 0)
        self.action_space = [(-1, 0), (1, 0), (0, -1), (0, 1)] # 0:U, 1:D, 2:L, 3:R

    def reset(self):
        self.agent_pos = (0, 0)
        self.done = False
        return np.array(self.agent_pos, dtype=np.float32)

    def step(self, action_idx):
        move = self.action_space[action_idx]
        next_pos = (self.agent_pos[0] + move[0], self.agent_pos[1] + move[1])
        
        # Obstacle/Edge Logic
        if (0 <= next_pos[0] < self.rows and 0 <= next_pos[1] < self.cols) and (next_pos not in self.obstacles):
            self.agent_pos = next_pos
            
        reward = 10 if self.agent_pos == self.terminal_state else -1
        done = self.agent_pos == self.terminal_state
        return np.array(self.agent_pos, dtype=np.float32), reward, done, {}

    def render(self):
        grid = np.full((self.rows, self.cols), "-")
        for obs in self.obstacles: grid[obs] = "X"
        grid[self.terminal_state] = "G"
        grid[self.agent_pos] = "O"
        return grid
