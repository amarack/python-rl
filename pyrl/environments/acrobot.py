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
class Acrobot(Environment):
    name = "Acrobot"

    def __init__(self, noise=0.0, reward_noise=0.0, random_start=False):
        self.noise = noise
        self.reward_noise = reward_noise
        self.random_start = random_start
        self.state = numpy.zeros((4,))
        self.domain_name = "Acrobot"

        maxes = numpy.array([numpy.pi, numpy.pi, 4.*numpy.pi, 9.*numpy.pi])
        self.state_range = numpy.array([-1.*maxes,maxes]).T

        self.goalPos = 1.0
        self.l1 = 1.0
        self.l2 = 1.0
        self.m1 = 1.0
        self.m2 = 1.0
        self.lc1 = 0.5
        self.lc2 = 0.5
        self.I1 = 1.0
        self.I2 = 1.0
        self.gravity = 9.8
        self.dt = 0.05
        self.simSteps = 4

    def makeTaskSpec(self):
        ts = TaskSpecRLGlue.TaskSpec(discount_factor=1.0, reward_range=(-1.0, 0.0))
        ts.setDiscountFactor(1.0)
        ts.addDiscreteAction((0, 2))
        for minValue, maxValue in self.state_range:
            ts.addContinuousObservation((minValue, maxValue))

        ts.setEpisodic()
        ts.setExtra(self.domain_name)
        return ts.toTaskSpec()

    def reset(self):
        if self.random_start:
            self.state = numpy.random.random((len(self.state_range),))
            self.state *= (self.state_range[:,1] - self.state_range[:,0]) + \
                self.state_range[:,0]
        else:
            self.state[:] = 0.0

    def env_init(self):
        return self.makeTaskSpec()

    def env_start(self):
        self.reset()
        returnObs = Observation()
        returnObs.doubleArray = self.state.tolist()
        return returnObs

    def isAtGoal(self):
        elbow_height = self.l1*numpy.cos(self.state[0])
        hand_height = elbow_height + self.l2*numpy.sin(numpy.pi/2.0 - self.state[:2].sum())
        #hand_height = elbow_height + self.l2*numpy.cos(self.state[0] + self.state[1])
        hand_height *= -1
        return hand_height > self.goalPos

    def takeAction(self, intAction):
        intAction -= 1.0
        actNoise = numpy.random.normal(scale=self.noise) if self.noise > 0 else 0.0
        intAction += actNoise

        for step in range(self.simSteps):
            d1 = self.m1 * pow(self.lc1, 2) + self.m2 * (pow(self.l1, 2) + pow(self.lc2, 2) + 2. * self.l1 * self.lc2 * numpy.cos(self.state[1])) + self.I1 + self.I2
            d2 = self.m2 * (pow(self.lc2, 2) + self.l1 * self.lc2 * numpy.cos(self.state[1])) + self.I2

            phi_2 = self.m2 * self.lc2 * self.gravity * numpy.cos(self.state[:2].sum() - numpy.pi / 2.0)
            phi_1 = -(self.m2 * self.l1 * self.lc2 * pow(self.state[3], 2) * numpy.sin(self.state[1]) - \
                      2. * self.m2 * self.l1 * self.lc2 * self.state[0] * self.state[1] * numpy.sin(self.state[1])) + \
                      (self.m1 * self.lc1 + self.m2 * self.l1) * self.gravity * numpy.cos(self.state[0] - numpy.pi / 2.0) + phi_2

            theta2_ddot = (intAction + (d2 / d1) * phi_1 - self.m2 * self.l1 * self.lc2 * pow(self.state[2], 2) * \
                       numpy.sin(self.state[1]) - phi_2) / (self.m2 * pow(self.lc2, 2) + self.I2 - pow(d2, 2) / d1)
            theta1_ddot = -(d2 * theta2_ddot + phi_1) / d1

            self.state[2] += theta1_ddot * self.dt
            self.state[3] += theta2_ddot * self.dt
            self.state[:2] += self.state[2:] * self.dt

        self.state = self.state.clip(self.state_range[:,0], self.state_range[:,1])

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
        theObs.doubleArray = self.state.tolist()

        returnRO = Reward_observation_terminal()
        returnRO.r = theReward
        returnRO.o = theObs
        returnRO.terminal = episodeOver

        return returnRO

    def env_cleanup(self):
        pass

    def env_message(self,inMessage):
        return "I don't know how to respond to your message";


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run Noisy Acrobot environment in network mode.')
    parser.add_argument("--noise", type=float, default=0, help="Standard deviation of additive noise to generate, affects the action effects.")
    parser.add_argument("--random_restarts", type=bool, default=False, help="Restart the state with random values.")

    args = parser.parse_args()
    EnvironmentLoader.loadEnvironment(Acrobot(noise=args.noise, random_start=args.random_restarts))

