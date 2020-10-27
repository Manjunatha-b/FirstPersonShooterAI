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
# action[1] = Move front back               (-1 = back, 0 = still,  1 = forward )
# action[2] = Shoot gun                     ( 0 = dont shoot,   1 = shoot )
# action[3] = Jump                          ( 0 = dont jump,    1 = jump )                              
# action[4] = Camera rotation horizontal    (-1 = left, 0 = still,  1 = right )
# action[5] = Camera rotation vertical      (-1 = down, 0 = still,  1 = up )
# action[6] = run                           ( 0 = dont run,     1 = run )
  
class Driver:
    def __init__(self,instanceCount):
        self.channel = EngineConfigurationChannel()
        self.env = UnityEnvironment(file_name="./ExecutableBuild/SupaHot",seed = 1,side_channels =[self.channel])
        self.behaviors = self.env.behavior_specs.items()
        self.channel.set_configuration_parameters(width=640,height=360)
        self.env.reset()
        for item in self.behaviors:
            self.name = item[0]
            self.obs_shape = item[1].observation_shapes
            self.action_shape = item[1].action_shape

        self.acts = [3,3,2,2,3]
        self.act_dict = {}
        self.tempset = set()
        self.count = 0
        self.GenerateActionTable(np.zeros(7),0)
        self.DoStep()

        self.instances = instanceCount
        self.policy = PPO(74,54,128,0.002,(0.9,0.999),0.99,2,0.1)
        self.memory = [Memory() for i in range(self.instances)]
    
    def DoStep(self):
        action = np.zeros((1,7))
        self.env.set_actions(self.name,action)
        self.env.step()

    ### this one has been cut-down to have 54 possible actions 
    def GenerateActionTable(self,array,curr):
        if(curr==3):
            self.GenerateActionTable(array,curr+1)
            return
        if(curr>=5):
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
        inp1 = np.asarray(state.obs[0])
        inp2 = np.asarray(state.obs[1])
        terminal = True if len(terminal.interrupted)!=0 else False
        state = np.concatenate((inp1,inp2),axis=1)
        state = torch.from_numpy(state).float().to('cpu') 
        return state, terminal

    def GetAction(self,states):

        action_vectors = []
        action_fwpasses = []
        action_logprobs = []

        for i in range(self.instances):
            action_fwpass,action_logprob,action_index = self.policy.policy_old.act(states[i],self.memory[i])
            action_vector = self.act_dict[action_index]
            action_vectors.append(action_vector)
            action_fwpasses.append(action_fwpass)
            action_logprobs.append(action_logprob)

        action_vectors = np.asarray(action_vectors).reshape((self.instances,self.action_shape))
        return action_vectors, action_fwpasses, action_logprobs
            
    def GetReward(self):
        stepinfo,_ = self.env.get_steps(self.name)
        return stepinfo.reward

    def write(self,actions,logprobs,states,next_states,rewards,terminal):
        for i in range(self.instances):
            self.memory[i].actions.append(action_fwpass[i])
            self.memory[i].logprobs.append(action_logprobs[i])
            self.memory[i].is_terminals.append(terminal)
            self.memory[i].states.append(states[i])
            self.memory[i].next_states.append(next_states[i])
            self.memory[i].rewards.append(rewards[i])
            self.memory[i].is_terminals.append(terminal)

driver = Driver(4)
states,_ = driver.GetObs()
running_reward = 0 
epno = 0
dets = open("log.txt",'w')

while(True):    
    ### Get Actions
    actions,action_fwpass,action_logprobs = driver.GetAction(states)

    ### Do the Actions
    driver.env.set_actions(driver.name,actions)
    driver.env.step()
    
    ### Get Reward
    rewards = driver.GetReward()

    ### Get Next State and End condition
    next_states,terminal = driver.GetObs()

    ### Write the state,reward,actions etc to the memory
    driver.write(action_fwpass,action_logprobs,states,next_states,rewards,terminal)

    ### current state becomes the previous step's next state
    states = next_states

    ### Add reward ( only for display )
    running_reward+=sum(rewards)

    if(terminal):
        epno+=1
        for i in range(driver.instances):
            driver.policy.update(driver.memory[i])
            driver.memory[i].clear_memory()
        if(epno%5==0):
            print("Episode: "+str(epno)+" Reward: "+str(running_reward/driver.instances))
            dets.write(str(running_reward/driver.instances)+"\n")
        running_reward=0
        


        