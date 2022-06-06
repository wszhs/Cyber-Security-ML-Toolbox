import random
from collections import defaultdict
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma

np.random.seed(31)
random.seed(31)

n_impressions = 10000
n_users = 100
n_items = 3

class User:
    def __init__(self, index):
        self.index = index
        self.interest_mean = np.random.randint(1, 10, n_items)
        
    def interest(self, index):
        return np.random.normal(self.interest_mean[index], 1)
        
        
class Item:
    def __init__(self, index):
        self.index = index
        self.score = np.random.randint(1, 10)
        
        
class Algorithm:
    
    def choose(self, user, items):
        raise NotImplemented

    def update(self, user, item):
        raise NotImplemented


def evaluate(algo, smooth=0.998):
    total_reward = 0
    smooth_reward_list = []
    prev_smooth_reward = 0
    for i in range(n_impressions):
        user = random.choice(users)
        item = algo.choose(user, items)
        reward = user.interest(item.index) * item.score
        algo.update(user, item, reward)

        # smooth reward for visualization
        smooth_reward = smooth * prev_smooth_reward + (1 - smooth) * reward  # exponential weighted average
        prev_smooth_reward = smooth_reward
        smooth_reward = smooth_reward / (1 - smooth ** (i + 1))  # bias correction
        smooth_reward_list.append(smooth_reward)
        
        total_reward += reward
    return smooth_reward_list, total_reward

users = [User(i) for i in range(n_users)]
items = [Item(i) for i in range(n_items)]

class EpsilonGreedy(Algorithm):
    
    def __init__(self, epsilon=0.1):
        self.init_epsilon = epsilon
        self.epsilon = epsilon
        
        self.values = defaultdict(lambda : defaultdict(int))
        self.counts = defaultdict(lambda : defaultdict(int))
        self.step = 0
    
    def choose(self, user, items):
        if np.random.uniform() < self.epsilon:
            item = random.choice(items)
        else:
            best_value = -float("inf")
            best_item = None
            for i in range(len(items)):
                value = self.values[user.index][i] / (self.counts[user.index][i] + 1e-4)
                if value > best_value:
                    best_value = value
                    best_item = items[i]
            item = best_item
        self.step += 1
        return item
    
    def update(self, user, item, reward):
        self.counts[user.index][item.index] += 1
        self.values[user.index][item.index] += reward
        self.epsilon -= (self.init_epsilon / n_impressions)
        
        
plt.figure(figsize=(14, 6))
for epsilon in [0, 0.12, 0.36, 0.48, 0.72, 1]:
    reward_list, info = evaluate(EpsilonGreedy(epsilon))
    plt.plot(reward_list, label=f"epsilon={epsilon}", alpha=0.5)
    print(f"epsilon: {epsilon}\t total_reward: {info:.2f}")
plt.grid()
plt.ylim((8, 18))
plt.legend()
plt.show()