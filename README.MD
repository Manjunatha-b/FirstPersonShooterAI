# FPS environment
<img> <img src="https://github.com/Manjunatha-b/FirstPersonShooterAI/blob/master/WorkingGif.gif" width="700"></img>
## Introduction

The aim of this project is to create a reinforcement learning platform for learning first person shooters and include functions to retrieve observations such as depth maps and object bounding boxes, and to test different ML algorithms on the same.

## Project So far

1. Implemented the first person camera using Unity assets.
1. Implemented mechanism to instantiate bullets on mousedown.
1. Implemented bullet collision using raycasts.
    1. The bullet returns to the main file - *RigidBodyFirstPersonController.cs* - after it hits an object with <b>"enemy"</b> tag, and destroys itself.
    1. The bullet doesnt return any value if it hits a non enemy tag, can be easily modified.
    1. The bullet destroys itself if it travels a distance of over 250f.
    1. Before destroying itself, the bullet spawns a particle system for explosion effect.
    1. the particle system automatically destroys itself on finishing its purpose.
1. Implemented a slowdown mechanism where the speed of time is proportional to your speed of movement in game ( not including camera ) ( Yes this is a SuperHot ripoff ).
1. Created simple enemy that doesn't move and gets destroyed by bullet.


## TODO :

1. Fast depth estimation algorithm using mobilenet for feature
1. Random spawning of enemy inside the game area
1. Train raw RL with only angle and position vectors
