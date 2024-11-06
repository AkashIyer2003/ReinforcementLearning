import numpy as np
import random

class EpsilonGreedyMAB:
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon  # Exploration probability
        self.counts = np.zeros(n_arms)  # Number of times each arm was chosen
        self.values = np.zeros(n_arms)  # Estimated success rates for each arm

    def select_arm(self):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_arms - 1)
        else:
            return np.argmax(self.values)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        current_value = self.values[chosen_arm]
        self.values[chosen_arm] = current_value + (1 / self.counts[chosen_arm]) * (reward - current_value)

class Environment:
    def __init__(self, n_arms):
        self.probabilities = np.random.rand(n_arms)

    def get_reward(self, arm):
        return 1 if random.random() < self.probabilities[arm] else 0

n_arms = 10          # Number of arms (recommendations)
n_rounds = 1000      # Number of rounds for the simulation
epsilon = 0.1        # Epsilon for exploration

env = Environment(n_arms)
mab_agent = EpsilonGreedyMAB(n_arms, epsilon)

total_rewards = 0

for round in range(n_rounds):
    chosen_arm = mab_agent.select_arm()
    
    reward = env.get_reward(chosen_arm)
    
    mab_agent.update(chosen_arm, reward)
    
    total_rewards += reward

print("Estimated success rates for each arm:", mab_agent.values)
print("True success probabilities for each arm:", env.probabilities)
print(f"Total reward after {n_rounds} rounds: {total_rewards}")
