import numpy as np
import random
from collections import defaultdict

class GridEnv:
    def __init__(self, grid_size, start, goal, obstacles):
        self.grid_size = grid_size
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.state = start  # current position of the agent
        self.actions = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1)    # right
        }

    def reset(self):
        self.state = self.start
        return self.state
    
    def step(self, action):
        x, y = self.state
        dx, dy = self.actions[action]  # Get the direction for the action
        new_x, new_y = x + dx, y + dy
        if 0 <= new_x < self.grid_size[0] and 0 <= new_y < self.grid_size[1]:
            new_state = (new_x, new_y)
        else:
            new_state = self.state
        if new_state in self.obstacles:
            reward = -1  # Negative reward for hitting an obstacle
            new_state = self.state  # Agent stays in the same position
        elif new_state == self.goal:
            reward = 1   # Positive reward for reaching the goal
            done = True  # The episode ends when the goal is reached
            return new_state, reward, done
        else:
            reward = 0  # Neutral reward for regular moves
        self.state = new_state
        done = (self.state == self.goal)
        return self.state, reward, done    

    def is_done(self):
        return self.state == self.goal

class RLAgent:
    def __init__(self, env, discount_factor=0.9, learning_rate=0.1, epsilon=0.1):
        self.env = env
        self.discount_factor = discount_factor  
        self.learning_rate = learning_rate      # Learning rate for Q-learning/SARSA
        self.epsilon = epsilon                  # Epsilon for epsilon-greedy exploration
        self.q_table = defaultdict(lambda: np.zeros(4))  # Q-table for storing Q-values for each (state, action)

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(4))  # Explore: choose a random action
        else:
            return np.argmax(self.q_table[state])  # Exploit: choose the best known action

    def value_iteration(self, theta=1e-4):
        V = defaultdict(float)  # State-value function
        while True:
            delta = 0
            for x in range(self.env.grid_size[0]):
                for y in range(self.env.grid_size[1]):
                    state = (x, y)
                    if state == self.env.goal or state in self.env.obstacles:
                        continue
                    v = V[state]
                    V[state] = max(
                        sum(
                            (1 * (reward + self.discount_factor * V[next_state]))
                            for reward, next_state in [self._next_state(state, action)]
                        )
                        for action in range(4)
                    )
                    delta = max(delta, abs(v - V[state]))
            if delta < theta:
                break
        for state in V:
            self.q_table[state] = np.array([
                sum(
                    (1 * (reward + self.discount_factor * V[next_state]))
                    for reward, next_state in [self._next_state(state, action)]
                )
                for action in range(4)
            ])
        print("Value Iteration Completed.")

    def q_learning(self, episodes=500):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                
                best_next_action = np.argmax(self.q_table[next_state])
                td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
                self.q_table[state][action] += self.learning_rate * (td_target - self.q_table[state][action])
                
                state = next_state
        print("Q-Learning Completed.")

    def sarsa(self, episodes=500):
        for episode in range(episodes):
            state = self.env.reset()
            action = self.choose_action(state)
            done = False
            while not done:
                next_state, reward, done = self.env.step(action)
                next_action = self.choose_action(next_state)
                
                # SARSA update rule
                td_target = reward + self.discount_factor * self.q_table[next_state][next_action]
                self.q_table[state][action] += self.learning_rate * (td_target - self.q_table[state][action])
                
                state = next_state
                action = next_action
        print("SARSA Completed.")

    def _next_state(self, state, action):
        original_state = self.env.state
        self.env.state = state
        next_state, reward, done = self.env.step(action)
        self.env.state = original_state
        return reward, next_state

if __name__ == "__main__":
    grid_size = (100, 100)
    start, goal = (0, 0), (99, 99)
    obstacles = set(random.sample([(x, y) for x in range(100) for y in range(100)], 1000))  # Random obstacles

    env = GridEnv(grid_size, start, goal, obstacles)
    agent = RLAgent(env)

    print("Starting Value Iteration...")
    agent.value_iteration()

    print("Starting Q-Learning...")
    agent.q_learning(episodes=500)

    print("Starting SARSA...")
    agent.sarsa(episodes=500)

