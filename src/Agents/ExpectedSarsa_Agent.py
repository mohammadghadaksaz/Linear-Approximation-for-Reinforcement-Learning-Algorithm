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