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
class Chain(Environment):
    """The simple 5-state chain domain often used in the literature for more
    theoretical methods that don't scale as well to large problems. Its also
    a good demonstration of the need for sufficient exploration.

    From paper:
    Bayesian Q-learning. 1998.
    Richard Dearden, Nir Friedman, and Stuart Russell.
    """
    name = "Chain"

    def __init__(self, **kwargs):
        self.state = 0
        self.chain_size = kwargs.setdefault("chain_size", 5)
        self.slip_prob = kwargs.setdefault("slip_prob", 0.2)
        self.goal_reward = 10.0
        self.left_reward = 2.0
        self.right_reward = 0.0

    def makeTaskSpec(self):
        ts = TaskSpecRLGlue.TaskSpec(discount_factor=0.99, reward_range=(0.0, 10.0))
        ts.addDiscreteAction((0, 1))
        ts.addDiscreteObservation((0, self.chain_size-1))
        ts.setContinuing()
        ts.setExtra(self.name)
        return ts.toTaskSpec()

    def getState(self):
        return [self.state]

    def reset(self):
        self.state = 0

    def env_init(self):
        return self.makeTaskSpec()

    def env_start(self):
        self.reset()
        returnObs = Observation()
        returnObs.intArray = self.getState()
        return returnObs

    def isAtGoal(self):
        return self.state == self.chain_size-1

    def takeAction(self, intAction):
        if numpy.random.random() < self.slip_prob:
            intAction = 0 if intAction == 1 else 1

        if intAction == 0:
            self.state = 0
            return self.left_reward
        else:
            self.state = min(self.chain_size-1, self.state+1)
            if self.isAtGoal():
                return self.goal_reward
            else:
                return self.right_reward

    def env_step(self,thisAction):
        intAction = int(thisAction.intArray[0])
        theReward = self.takeAction(intAction)
        theObs = Observation()
        theObs.intArray = self.getState()

        returnRO = Reward_observation_terminal()
        returnRO.r = theReward
        returnRO.o = theObs
        returnRO.terminal = 0

        return returnRO

    def env_cleanup(self):
        pass

    def env_message(self,inMessage):
        return "I don't know how to respond to your message";




