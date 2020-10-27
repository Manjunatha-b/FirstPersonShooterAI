import torch
import torch.nn as nn

class IntrinsicCuriosity(nn.Module):
    def __init__(self,action_dim,state_dim,feature_dim):
        super(IntrinsicCuriosity,self).__init__()


        self.feature_model = nn.Sequential(
            nn.Linear(state_dim,feature_dim),
        )

        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim+action_dim,feature_dim)
        )

        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim,action_dim),
        )
    
    def forward(self,state,next_state,action):
        state_ft = self.feature_model(state)
        next_state_ft = self.feature_model(next_state)
        
        inv_inp = torch.cat((state_ft,next_state_ft),1)
        for_inp = torch.cat((state_ft,action),1)

        inv_op = self.inverse_model(inv_inp)
        for_op = self.forward_model(for_inp)

        return inv_op,for_op


icm = IntrinsicCuriosity()
state = torch.rand(75)
next_state = torch.rand(75)
