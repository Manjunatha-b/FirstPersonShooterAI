import numpy as np
import sys
sys.path.insert(1,"./MLFiles")
from MLFiles.DQN import DQN
from MLFiles.DQN import ReplayBuffer
import torch.optim as optim
from torch.autograd import Variable
import torch
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

# action[0] = Move left right               (-1 = left, 0 = still,  1 = right )
# action[1] = Move front back               (-1 = back, 0 = still,  1 = forward)
# action[2] = Shoot gun                     ( 0 = dont shoot,   1 = shoot)
# action[3] = Jump                          ( 0 = dont jump,    1 = jump)                              
# action[4] = Camera rotation horizontal    (-1 = left, 0 = still,  1 = right)
# action[5] = Camera rotation vertical      (-1 = down, 0 = still,  1 = up)
# action[6] = run                           ( 0 = dont run,     1 = run)  

class Driver:
    def __init__(self):
        self.channel = EngineConfigurationChannel()
        self.env = UnityEnvironment(file_name="./ExecutableBuild/SupaHot",seed = 1,side_channels =[self.channel],base_port=5004)
        self.behaviors = self.env.behavior_specs.items()
        self.channel.set_configuration_parameters(width=640,height=360)
        self.env.reset()
        for item in self.behaviors:
            self.name = item[0]
            self.obs_shape = item[1].observation_shapes
            self.action_shape = item[1].action_shape

        self.acts = [3,3,2,2,3,3,2]
        self.act_dict = {}
        self.tempset = set()
        self.count = 0
        self.GenerateActionTable(np.zeros(7),0)

        self.dqn = DQN(77,len(self.act_dict),1000)

        self.optimizer = optim.Adam(self.dqn.parameters())
        self.replay = ReplayBuffer(1000)
        self.gamma = 0.95
        

    def generic_action(self):
        action = np.zeros((1,7))
        action[0][5] = -1
        self.env.set_actions(self.name,action)
        return self.env.step()

    def GenerateActionTable(self,array,curr):
        if(curr>=7):
            return 
        for i in range(self.acts[curr]):
            array[curr] = 1-i
            if(tuple(array) not in self.tempset):
                self.act_dict[self.count] = np.asarray(array).reshape(1,7)
                self.count+=1
                self.tempset.add(tuple(array))
            self.GenerateActionTable(list(array),curr+1)
    
    def GetObs(self):
        state,terminal = self.env.get_steps(self.name)
        inp1 = np.asarray(state.obs[0][0])
        inp2 = np.asarray(state.obs[1][0])
        terminal = True if len(terminal.interrupted)==1 else False
        state = np.concatenate((inp1,inp2))
        return state, terminal

    def GetAction(self,state):
        action_index = self.dqn.act(state)
        action = self.act_dict[action_index]
        return action, action_index
    
    def DoStep(self,action):
        self.env.set_actions(self.name,action)
        self.env.step()

    def GetReward(self):
        stepinfo,_ = self.env.get_steps(self.name)
        return stepinfo.reward[0]

    def TD_loss(self,batch_size):
        state,action,reward,next_state,terminal = self.replay.sample(batch_size)

        state      = Variable(torch.FloatTensor(np.float32(state)))
        next_state = Variable(torch.FloatTensor(np.float32(next_state)), requires_grad=False)
        action     = Variable(torch.LongTensor(np.float32(action)))
        reward     = Variable(torch.FloatTensor(reward))
        done       = Variable(torch.FloatTensor(terminal))

        q_values      = self.dqn(state)
        next_q_values = self.dqn(next_state)
        
        q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value     = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)
        
        loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss


driver = Driver()
running_reward = 0 
dets = open("log.txt",'w')

while(True):
    terminal = False

    curr_state, _= driver.GetObs()  
    action, action_index = driver.GetAction(curr_state)
    driver.DoStep(action)
    next_state, terminal = driver.GetObs()
    reward = driver.GetReward()
    driver.replay.push(curr_state,np.float32(action_index),np.float32(reward),next_state,terminal)
    running_reward+=reward

    if(terminal):
        loss = driver.TD_loss(256)
        print(running_reward)
        dets.write(str(running_reward)+"\n")
        running_reward=0
        

driver.env.close()
dets.close()



