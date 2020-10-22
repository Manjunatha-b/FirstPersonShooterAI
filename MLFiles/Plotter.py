import matplotlib.pyplot as plt
import numpy as np

file = open("./Logs/ppo1.txt",'r')
rewards = []
for item in file:
    bruh = file.readline()
    if(not bruh==""):
        rewards.append(float(bruh))
print(len(rewards))
plt.plot(rewards)
plt.show()