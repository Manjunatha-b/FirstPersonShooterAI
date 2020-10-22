import numpy as np
import sys
sys.path.insert(1,"./MLFiles")
from MLFiles.PPO import PPO
from MLFiles.PPO import Memory
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
        self.env = UnityEnvironment(file_name="./ExecutableBuild/SupaHot",seed = 1,side_channels =[self.channel])
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

        self.ppo = PPO(77,len(self.act_dict),1024,0.002,(0.9,0.999),0.99,2,0.1)
        self.memory = Memory()
        


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
        action_index = self.ppo.policy_old.act(state,self.memory)
        action = self.act_dict[action_index]
        return action
    
    def DoStep(self,action):
        self.env.set_actions(self.name,action)
        self.env.step()

    def GetReward(self):
        stepinfo,_ = self.env.get_steps(self.name)
        return stepinfo.reward[0]


driver = Driver()
running_reward = 0 
dets = open("log.txt",'w')
epno = 0

# while(True):
#     terminal = False

#     curr_state, _= driver.GetObs()  
#     action = driver.GetAction(curr_state)
#     driver.DoStep(action)
#     next_state, terminal = driver.GetObs()
#     reward = driver.GetReward()
#     driver.memory.rewards.append(reward)
#     driver.memory.is_terminals.append(terminal)
#     running_reward+=reward

#     if(terminal):
#         epno+=1
#         driver.ppo.update(driver.memory)
#         driver.memory.clear_memory()
#         print("Episode: "+str(epno)+" Running Reward: "+str(running_reward))
#         dets.write(str(running_reward)+"\n")
#         running_reward=0
#         torch.save(driver.ppo.policy.state_dict(),"PPOBrain.pth")

# dets.close()