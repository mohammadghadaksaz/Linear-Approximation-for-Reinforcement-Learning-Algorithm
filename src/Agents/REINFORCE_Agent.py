import numpy as np
import random


class REINFORCE:
    def __init__(self, env, alpha, temp, temp_dec, temp_stop, gamma,
                  num_tiles=150, num_bins=10):           
        self.env = env
        self.alpha = alpha
        self.temp = temp
        self.temp_dec = temp_dec
        self.temp_stop = temp_stop
        self.gamma = gamma
        self.num_tiles = num_tiles
        self.num_bins = num_bins
        self.theta = np.random.uniform(-0.0001, 0.0001,
         (self.env.action_space.n,  num_tiles * num_bins 
                        * env.observation_space.shape[0]))