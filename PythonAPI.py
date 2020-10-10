import numpy as np
import sys
sys.path.insert(1,"./MLFiles")
import MLFiles.DQN
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

    def generic_action(self):
        action = np.zeros((1,7))
        action[0][5] = -1
        self.env.set_actions(self.name,action)
        self.env.step()

    def GenerateActionTable(self,array,curr):
        if(curr>=7):
            return 
        for i in range(self.acts[curr]):
            array[curr] = 1-i
            if(tuple(array) not in self.tempset):
                self.act_dict[self.count] = array
                self.count+=1
                self.tempset.add(tuple(array))
            self.GenerateActionTable(list(array),curr+1)

driver = Driver()




