#
# Copyright (C) 2013, Will Dabney
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy
from rlglue.environment.Environment import Environment
from rlglue.environment import EnvironmentLoader as EnvironmentLoader
from rlglue.types import Observation
from rlglue.types import Action
from rlglue.types import Reward_observation_terminal
from pyrl.rlglue import TaskSpecRLGlue
from pyrl.rlglue.registry import register_environment

@register_environment
class MarbleMaze(Environment):
    """A simple gridworld like domain, with many wall segments to create
    a maze. This domain, unlike the other gridworld domains will be made
    entirely discrete.

    From paper:
    A Bayesian Sampling Approach to Exploration in Reinforcement Learning. 2009.
    John Asmuth, Lihong Li, Michael Littman, Ali Nouri, and David Wingate.
    """
    name = "Marble Maze"

    def __init__(self, **kwargs):
        # Building walls, each int in the maze matrix represents the type of wall setup
        # 0000 no walls
        # 0001 wall to the north
        N = 1
        # 0010 wall to the east
        E = 2
        # 0100 wall to the south
        S = 4
        # 1000 wall to the west
        W = 8
        self.directions = numpy.array([N,E,S,W], dtype=int)
        self.maze = numpy.array([[N+W, N, N, N, N, N+E],
                                [W+S, N+E+S, W, E, W, E],
                                [N+W, N+S, S, S+E, W, E],
                                [S+W, N+S, N, N, E, E],
                                [N+W, N, 0, S, 0, E],
                                [S+W, S, S+E, S+W+N, S+E, S+E+W]], dtype=int)
        self.pits = numpy.array([[1,1], [4,1], [4,2], [3, 3]], dtype=int)
        self.noise = kwargs.setdefault('noise', 0.2)
        self.start_loc = numpy.zeros((2,), dtype=int)
        self.pos = numpy.zeros((2,), dtype=int)
        self.step_reward = -0.001
        self.goal_loc = numpy.array([5,5], dtype=int)
        self.domain_name = "Marbel Maze (Discrete)"

    def makeTaskSpec(self):
        ts = TaskSpecRLGlue.TaskSpec(discount_factor=0.95, reward_range=(-1.0, 1.0))
        ts.addDiscreteAction((0, 3))
        ts.addDiscreteObservation((0, self.maze.shape[0]-1))
        ts.addDiscreteObservation((0, self.maze.shape[1]-1))
        ts.setEpisodic()
        ts.setExtra(self.domain_name)
        return ts.toTaskSpec()

    def getState(self):
        return self.pos.tolist()

    def reset(self):
        self.pos = self.start_loc.copy()

    def env_init(self):
        return self.makeTaskSpec()

    def env_start(self):
        self.reset()
        returnObs = Observation()
        returnObs.intArray = self.getState()
        return returnObs

    def isAtGoal(self):
        return (self.pos == self.goal_loc).all()

    def takeAction(self, intAction):
        direction = numpy.zeros((2,), dtype=int)
        direction[int(intAction)/2] = 1 + (intAction % 2)*-2

        # Noisy movement causes agent to move perpendicular to
        # the desired action, with equal likelihood for either option
        if numpy.random.random() < self.noise:
            direction.fill(0)
            direction[int(intAction < 2)] = numpy.random.randint(2)*-2 + 1
        if self.maze[tuple(self.pos)] % 2 != 0: # North wall
            direction[0] = max(0, direction[0])
        if (self.maze[tuple(self.pos)] % 8 >= 4): # South wall
            direction[0] = min(0, direction[0])
        if (self.maze[tuple(self.pos)] % 4 >= 2): # East wall
            direction[1] = min(0, direction[1])
        if self.maze[tuple(self.pos)] >= 8: # West wall
            direction[1] = max(0, direction[1])

        self.pos += direction
        if self.isAtGoal():
            return 1.0, True
        elif self.pos.tolist() in self.pits.tolist():
            return -1.0, True
        else:
            return self.step_reward, False

    def env_step(self,thisAction):
        intAction = int(thisAction.intArray[0])
        theReward, episodeOver = self.takeAction(intAction)

        theObs = Observation()
        theObs.intArray = self.getState()

        returnRO = Reward_observation_terminal()
        returnRO.r = theReward
        returnRO.o = theObs
        returnRO.terminal = int(episodeOver)

        return returnRO

    def env_cleanup(self):
        pass

    def env_message(self,inMessage):
        return "I don't know how to respond to your message";




