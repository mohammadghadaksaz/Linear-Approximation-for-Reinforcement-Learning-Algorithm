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
        
    def softmax(self, state):
        preferences = np.exp(np.sum((self.theta * state) / self.temp, axis=1))
        return preferences / np.sum(preferences)
    
    def choose_action(self, state):
        action_probs = self.softmax(state)
        actions = np.array(range(self.env.action_space.n))
        return np.random.choice(actions, p=action_probs)
    
    def update(self, state, G, action):
        Qs = np.sum(np.exp(np.sum((self.theta * state) / self.temp, axis=1)))
        self.theta[action] += self.alpha * G * \
            (state / self.temp - state / (Qs * self.temp) \
             * np.exp(np.sum(self.theta[action] * state) / self.temp) )

    def temp_decay(self):
        if self.temp > self.temp_stop:
            self.temp *= self.temp_dec
