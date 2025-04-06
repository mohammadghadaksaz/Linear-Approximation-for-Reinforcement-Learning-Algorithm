import numpy as np
import random

class ActorCritic:
    def __init__(self, env, alpha_weights, alpha_thetas, temp, temp_dec,
                  temp_stop, gamma, num_tiles=150, num_bins=10):           
        self.env = env
        self.alpha_weights = alpha_weights
        self.alpha_thetas = alpha_thetas
        self.temp = temp
        self.temp_dec = temp_dec
        self.temp_stop = temp_stop
        self.gamma = gamma
        self.num_tiles = num_tiles
        self.num_bins = num_bins
        self.theta = np.random.uniform(-0.0001, 0.0001, (self.env.action_space.n,
                      num_tiles * num_bins * env.observation_space.shape[0]))
        self.weights = np.random.uniform(-0.0001, 0.0001, 
                    (num_tiles * num_bins * env.observation_space.shape[0]))
        self.I = 1
    
    def softmax(self, state):
        preferences = np.exp(np.sum((self.theta * state) / self.temp, axis=1))
        return preferences / np.sum(preferences)
    
    def choose_action(self, state):
        action_probs = self.softmax(state)
        actions = np.array(range(self.env.action_space.n))
        return np.random.choice(actions, p=action_probs)
    
    def update(self, state, action, reward, next_state):
        delta = reward + self.gamma*np.sum(self.weights * next_state) - np.sum(self.weights*state)
        self.weights += self.alpha_weights*delta*state
        Qs = np.sum(np.exp(np.sum((self.theta * state) / self.temp, axis=1)))
        self.theta[action] += self.alpha_thetas * self.I * delta *\
     (state / self.temp - state / (Qs * self.temp) * np.exp(np.sum(self.theta[action] 
                                                             * state) / self.temp))
        self.I = self.gamma*self.I
        
    def temp_decay(self):
        if self.temp > self.temp_stop:
            self.temp *= self.temp_dec