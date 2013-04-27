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

import sys
import numpy

from rlglue.environment.Environment import Environment
from rlglue.environment import EnvironmentLoader as EnvironmentLoader
from rlglue.types import Observation
from rlglue.types import Action
from rlglue.types import Reward_observation_terminal

from pyrl.rlglue import TaskSpecRLGlue
from pyrl.rlglue.registry import register_environment

from . import gridworld
from scipy.stats import norm

@register_environment
class FuelWorld(gridworld.Gridworld):
	name = "Fuel World"

	# This is a continuous version of Todd Hester's Fuel World domain. 
	# As such, we will make the size, starting locations, and goal fixed to 
	# match the original's specifications. We will keep the additive gaussian noise, 
	# and as mentioned this will be continuous instead of discrete state spaces.
	def __init__(self, noise=0.0, fudge=1.4143, variation=(-10.0, -13.0, 5.0), fuel_noise=0.0):
		gridworld.Gridworld.__init__(self, size_x=31.0, size_y=21.0, goal_x=24.0, goal_y=11.0, 
					     noise=noise, random_start=True, fudge=fudge)
		self.fuel = 0.0
		self.fuel_noise = fuel_noise
		self.var = variation
		self.domain_name = "Continuous Fuel World"
		

	def makeTaskSpec(self):
		ts = TaskSpecRLGlue.TaskSpec(discount_factor=1.0, reward_range=(-400.0, 0.0))
		ts.addDiscreteAction((0, 7))
		ts.addContinuousObservation((0.0, self.size[0]-1))
		ts.addContinuousObservation((0.0, self.size[1]-1))
		ts.addContinuousObservation((-1.0, 60.0)) # Fuel range as per FuelRooms.cc
		ts.setEpisodic()
		ts.setExtra(self.domain_name)
		return ts.toTaskSpec()

	def env_start(self):
		self.reset()
		returnObs = Observation()
		returnObs.doubleArray = self.pos.tolist() + [self.fuel]
		return returnObs

	def reset(self):
		# Randomly start in the rectangle around (0,7),(4,12)
		self.pos = numpy.random.random((2,))
		self.pos[0] *= 4.0
		self.pos[1] *= 5.0
		self.pos[1] += 7.0

		self.fuel = numpy.random.random()*4.0 + 14.0 # Between 14 and 18

	def inFuelCell(self, position):
		return self.pos[1] <= 1.0 or self.pos[1] >= self.size[1]-1.0

	def isAtGoal(self):
		return gridworld.Gridworld.isAtGoal(self) or self.fuel < 0

	def getState(self):
		return gridworld.Gridworld.getState(self) + [self.fuel]

	def takeAction(self, intAction):
		if intAction == 0:
			self.pos[0] += 1.0
		elif intAction == 1:
			self.pos[0] -= 1.0
		elif intAction == 2:
			self.pos[1] += 1.0
		elif intAction == 3:
			self.pos[1] -= 1.0
		elif intAction == 4:
			self.pos += numpy.array([-1.0, 1.0])
		elif intAction == 5:
			self.pos += numpy.array([1.0, 1.0])
		elif intAction == 6:
			self.pos += numpy.array([-1.0, -1.0])
		elif intAction == 7:
			self.pos += numpy.array([1.0, -1.0])

		if self.noise > 0:
			self.pos += numpy.random.normal(scale=self.noise, size=(2,))

		self.pos = self.pos.clip([0, 0], self.size)

		self.fuel -= 1.0
		if self.fuel_noise > 0:
			self.fuel += numpy.random.normal(scale=self.fuel_noise)
		
		if self.inFuelCell(self.pos):
			self.fuel += 20.0
		if self.fuel > 60.0:
			self.fuel = 60.0

		if gridworld.Gridworld.isAtGoal(self):
			return 0.0
		elif self.fuel < 0:
			return -400.0
		elif self.inFuelCell(self.pos): # Fuel costs
			base = self.var[0] if self.pos[1] <= 1.0 else self.var[1]
			a = self.var[2]
			return base - (int(self.pos[0]) % 5)*a
		elif intAction < 4:
			return -1.0
		elif intAction >= 4:
			return -1.4
		else:
			print "ERROR in FuelWorld.takeAction"


if __name__=="__main__":
	import argparse
	parser = argparse.ArgumentParser(description='Run 2D MultiRoom Noisy Continuous Gridworld environment in network mode.')
	gridworld.addGridworldArgs(parser)
	parser.add_argument("--fuel_noise", type=float, default=0.0, 
			    help="If non-zero then gives the standard deviation of the additive Gaussian noise to add to the fuel expenditure.")
	args = parser.parse_args()
	EnvironmentLoader.loadEnvironment(FuelWorld(noise=args.noise, fudge=args.fudge, fuel_noise=args.fuel_noise))
