import numpy as np
import random


class ExpectedSarsa:
    def __init__(self, env, alpha, epsilon, gamma, num_tiles=3,
                  num_bins=10):
        self.env = env
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.num_tiles = num_tiles
        self.num_bins = num_bins
        self.theta = np.random.uniform(-0.001, 0.001, (self.env.action_space.n,
                        num_tiles * num_bins * env.observation_space.shape[0]))
        
    def choose_action(self, state):
        if (random.random() < self.epsilon):
            action = random.randint(0, self.env.action_space.n - 1)
        else:
            action = np.argmax(np.sum(self.theta * state, axis=1))
        return action

    def update(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            Q = np.sum(self.theta * state, axis=1)
            probabilities = [self.epsilon if i == np.argmax(Q) else
                   (1 - self.epsilon)/(self.env.action_space.n - 1)
                      for i in range(self.env.action_space.n)]
            target = reward + self.gamma * np.sum(probabilities * 
                        (np.sum(self.theta * next_state, axis=1)))
        self.theta[action] += self.alpha * (target - 
                    np.sum(self.theta[action] * state)) * state