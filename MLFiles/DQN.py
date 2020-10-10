import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):

    def __init__(self,inp_dim,out_dim,nodes):
        super(DQN,self).__init__()

        self.model = nn.Sequential(
            nn.Linear(inp_dim,nodes),
            nn.Tanh(),
            nn.Linear(nodes,out_dim)
        )
        self.learning_rate = 0.01
        self.gamma = 0.95
    
    def forward(self):
        raise NotImplementedError

    def act(self,observations):
        action = self.model(observations)
        return action


bruh = DQN(10,7,20)
bruh = bruh.float()
testinp = torch.from_numpy(np.ones((1,10))).float()

