import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

class DQN(nn.Module):

    def __init__(self,inp_dim,out_dim,nodes):
        super(DQN,self).__init__()

        self.model = nn.Sequential(
            nn.Linear(inp_dim,nodes),
            nn.Tanh(),
            nn.Linear(nodes,out_dim),
            nn.Softmax(dim =-1)
        )
        self.learning_rate = 0.01
        self.gamma = 0.95
    
    def forward(self):
        raise NotImplementedError

    def act(self,observations):
        observations = torch.from_numpy(observations).float()
        action = self.model(observations)
        action = Categorical(action)
        action = action.sample()
        return action.item()


