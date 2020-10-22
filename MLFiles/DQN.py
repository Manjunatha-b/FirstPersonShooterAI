import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.autograd import Variable
from collections import deque
import numpy as np
import random 

class DQN(nn.Module):

    def __init__(self,inp_dim,out_dim,nodes):
        super(DQN,self).__init__()

        self.model = nn.Sequential(
            nn.Linear(inp_dim,nodes),
            nn.ReLU(),
            nn.Linear(nodes,out_dim),
            nn.Softmax(dim =-1)
        )

        self.epsilon = 0.7
        self.inp_dim = inp_dim
        self.out_dim = out_dim

    def forward(self,state):
        return self.model(state)

    def act(self,state):
        if random.random() > self.epsilon:
            state   = Variable(torch.FloatTensor(state).unsqueeze(0), requires_grad=False)
            q_value = self.model(state)
            action  = q_value.max(1)[1].data[0].item()
        else:
            action = random.randrange(self.out_dim)
        return action


class ReplayBuffer:
    def __init__(self,capacity):
        self.buffer = deque(maxlen = capacity)

    def push(self,state,action,reward,new_state,terminal):
        self.buffer.append((state,action,reward,new_state,terminal))

    def sample(self,batch_size):
        states,actions,rewards,new_states,terminals = zip(*random.sample(self.buffer, batch_size))
        return states,actions,rewards,new_states,terminals

dqn = DQN(77,648,1000)
ret = dqn.act(np.zeros(77))
print(ret)