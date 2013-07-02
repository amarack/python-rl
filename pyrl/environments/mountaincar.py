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
class MountainCarND(Environment,object):
    """A generalized Mountain Car domain, which allows N-dimensional
    movement. When dimension=2 this behaves exactly as the classical
    Mountain Car domain. For dimension=3 it behaves as given in the
    paper:

    Autonomous Transfer for Reinforcement Learning. 2008.
    Matthew Taylor, Gregory Kuhlmann, and Peter Stone.

    However, this class also allows for even greater dimensions.
    """
    name = "3D Mountain Car"
    def __init__(self, **kwargs):#noise=0.0, random_start=False, dim=2):
        dim = int(max(2, kwargs.setdefault('dimension', 3)))
        self.noise = kwargs.setdefault('noise', 0.0)
        self.reward_noise = kwargs.setdefault('reward_noise', 0.0)
        self.random_start = kwargs.setdefault('random_start', False)
        self.state = numpy.zeros((dim-1,2))
        self.state_range = numpy.array([[[-1.2, 0.6], [-0.07, 0.07]] for i in range(dim-1)])
        self.goalPos = 0.5
        self.acc = 0.001
        self.gravity = -0.0025
        self.hillFreq = 3.0
        self.delta_time = 1.0

    def makeTaskSpec(self):
        ts = TaskSpecRLGlue.TaskSpec(discount_factor=1.0, reward_range=(-1.0, 0.0))
        ts.setDiscountFactor(1.0)
        ts.addDiscreteAction((0, self.state.shape[0]*2))
        flattened_ranges = self.state_range.reshape((numpy.prod(self.state_range.shape[:2]), self.state_range.shape[2]))
        for minValue, maxValue in flattened_ranges:
            ts.addContinuousObservation((minValue, maxValue))
        ts.setEpisodic()
        ts.setExtra(self.name)
        return ts.toTaskSpec()

    def reset(self):
        if self.random_start:
            self.state = numpy.random.random(self.state.shape)
            self.state *= (self.state_range[:,:,1] - self.state_range[:,:,0])
            self.state += self.state_range[:,:,0]
        else:
            self.state = numpy.zeros(self.state.shape)
            self.state[:,0] = -0.5

    def env_init(self):
        return self.makeTaskSpec()

    def env_start(self):
        self.reset()
        returnObs = Observation()
        returnObs.doubleArray = self.state.flatten().tolist()
        return returnObs

    def isAtGoal(self):
        return (self.state[:,0] >= self.goalPos).all()

    def takeAction(self, intAction):
        # Translate action into a (possibly) multi-dimensional direction
        intAction -= 1
        direction = numpy.zeros((self.state.shape[0],)) # Zero is Neutral
        if intAction >= 0:
            direction[int(intAction)/2] = ((intAction % 2) - 0.5)*2.0
        if self.noise > 0:
            direction += self.acc * numpy.random.normal(scale=self.noise, size=direction.shape)

        self.state[:,1] += self.acc*(direction) + self.gravity*numpy.cos(self.hillFreq*self.state[:,0])
        self.state[:,1] = self.state[:,1].clip(min=self.state_range[:,1,0], max=self.state_range[:,1,1])
        self.state[:,0] += self.delta_time * self.state[:,1]
        self.state[:,0] = self.state[:,0].clip(min=self.state_range[:,0,0], max=self.state_range[:,0,1])

    def env_step(self,thisAction):
        episodeOver = 0
        theReward = -1.0
        intAction = thisAction.intArray[0]

        self.takeAction(intAction)

        if self.isAtGoal():
            theReward = 0.0
            episodeOver = 1

        if self.reward_noise > 0:
            theReward += numpy.random.normal(scale=self.reward_noise)

        theObs = Observation()
        theObs.doubleArray = self.state.flatten().tolist()

        returnRO = Reward_observation_terminal()
        returnRO.r = theReward
        returnRO.o = theObs
        returnRO.terminal = episodeOver

        return returnRO

    def env_cleanup(self):
        pass

    def env_message(self,inMessage):
        return "I don't know how to respond to your message";

@register_environment
class MountainCar(MountainCarND):
    name = "Mountain Car"

    def __init__(self, **kwargs):
        kwargs['dimension'] = 2
        super(MountainCar, self).__init__(**kwargs)


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run Noisy Mountain Car environment in network mode.')
    parser.add_argument("--noise", type=float, default=0, help="Standard deviation of additive noise to generate, affects the action effects.")
    parser.add_argument("--random_restarts", type=bool, default=False, help="Restart the cart with a random location and velocity.")

    args = parser.parse_args()
    EnvironmentLoader.loadEnvironment(MountainCar(noise=args.noise, random_start=args.random_restarts))

