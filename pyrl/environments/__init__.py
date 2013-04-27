
# Author: Pierre-Luc Bacon <pierrelucbacon@gmail.com>

__all__ = ["fuelworld", "gridworld", "mountaincar", "acrobot", "cartpole", 
           "multiroom", "skeleton_environment", "taxi", "windyworld", 
           "batch_replenish"]

try:
    import pyrl.environments.generic_pomdp
    __all__.append("generic_pomdp")
except:
    pass


try:
    import pygame
    __all__.append("pinball")
except:
    pass
