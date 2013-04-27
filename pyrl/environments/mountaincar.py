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
class MountainCar(Environment):
	name = "Mountain Car"

	def __init__(self, noise=0.0, random_start=False):
            self.noise = noise
            self.random_start = random_start
            self.state = numpy.zeros((2,))
            self.domain_name = "Mountain Car"

            self.state_range = numpy.array([[-1.2, 0.6], [-0.07, 0.07]])
            self.goalPos = 0.5
            self.delta_time = 1.0
            self.acc = 0.001
            self.gravity = -0.0025
            self.hillFreq = 3.0

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
                    self.state = numpy.random.random((2,)) 
                    self.state *= (self.state_range[:,1] - self.state_range[:,0]) + \
                        self.state_range[:,0]
		else:
                    self.state = numpy.array([-0.5, 0.0])

	def env_init(self):
		return self.makeTaskSpec()
	
	def env_start(self):
            self.reset()
            returnObs = Observation()
            returnObs.doubleArray = self.state.tolist()
            return returnObs
		
	def isAtGoal(self):
            return self.state[0] >= self.goalPos

        def takeAction(self, intAction):
            intAction -= 1
            actNoise = self.acc*numpy.random.normal(scale=self.noise) if self.noise > 0 else 0.0
            self.state[1] += self.acc*(actNoise + intAction) + self.gravity*numpy.cos(self.hillFreq*self.state[0])
            self.state[1] = max(self.state_range[1,0], min(self.state_range[1,1], self.state[1]))

            self.state[0] += self.delta_time * self.state[1]
            self.state[0] = max(self.state_range[0,0], min(self.state_range[0,1], self.state[0]))

	def env_step(self,thisAction):
		episodeOver = 0
		theReward = -1.0
		intAction = thisAction.intArray[0]
		
		self.takeAction(intAction)

		if self.isAtGoal():
			theReward = 0.0
			episodeOver = 1

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
	parser = argparse.ArgumentParser(description='Run Noisy Mountain Car environment in network mode.')
	parser.add_argument("--noise", type=float, default=0, help="Standard deviation of additive noise to generate, affects the action effects.")
	parser.add_argument("--random_restarts", type=bool, default=False, help="Restart the cart with a random location and velocity.")

	args = parser.parse_args()
	EnvironmentLoader.loadEnvironment(MountainCar(noise=args.noise, random_start=args.random_restarts))

