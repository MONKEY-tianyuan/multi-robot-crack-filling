import os
import numpy as np
import torch as T
import torch.optim as optim
from torch.distributions.categorical import Categorical
import torch.nn as nn


class PPOMemory:
    def __init__(self,
                 batch_size) -> None:
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        
        self.batch_size = batch_size
        
    def genrate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0,n_states,self.batch_size)
        indices = np.arange(n_states,dtype=np.int64)
        np.random.shuffle(indices)
        
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        return np.array(self.states),\
            np.array(self.actions),\
                np.array(self.probs),\
                    np.array(self.vals),\
                        np.array(self.rewards),\
                            np.array(self.dones),\
                                batches
    
    def store_memory(self,state,action,prob,val,reward,done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)
        
    def clear_memory(self):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
    
    