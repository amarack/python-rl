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
#from rlglue.utils import TaskSpecVRLGLUE3
from pyrl.rlglue import TaskSpecRLGlue

class MultiRoomGridworld(Environment):

	# All parameters are in units of 1, where 1 is how far on average
	# the agent can move with a single action.
	# The walls will always be of unit thickness and be placed 
	# at 0.5*size_y with a door at 0.9*size_x, and 
	# above that wall a vertical wall will be placed at 0.3*size_x with a door at 0.75*size_y
	# If the goal falls inside a wall it will be pushed to the nearest non-wall location
	def __init__(self, size_x, size_y, goal_x, goal_y, noise=0.0, random_start=False, fudge=1.4143):
		self.size = numpy.array([size_x, size_y])

		# Build walls and doors (actually might only need to specify the doors)
		#self.wall1 = numpy.array([[0.0, size_y*0.5], [size_x, size_y*0.5]])
		self.door1 = numpy.array([size_x*0.9, size_y*0.5])
		#self.wall2 = numpy.array([[size_x*0.3, size_y*0.5], [size_x*0.3, size_y]])
		self.door2 = numpy.array([size_x*0.3, size_y*0.75])

		self.goal = self.fixPoint(numpy.array([goal_x, goal_y]))
		self.noise = noise
		self.random_start = random_start
		self.pos = numpy.zeros((2,))
		self.fudge = fudge

	def fixPoint(self, point):
		if numpy.abs(self.door1 - point).max() <= 0.5 or numpy.abs(self.door2 - point).max() <= 0.5:
			return point

		cond1 = point[1] <= self.door1[1]
		cond2 = point[0] <= self.door2[0]
		
		if cond1: # Bottom room
			return point.clip([0.0, 0.0], [self.size[0], self.door1[1]-0.51])
		else:
			if cond2: # Top left room
				return point.clip([0.0, self.door1[1]+0.51], [self.door2[0]-0.51, self.size[1]])
			else: # Top right room
				return point.clip([self.door2[0]+0.51, self.door1[1]+0.51], self.size)

	def isPointInWall(self, point):
		if (self.fixPoint(point) == point).all():
			return False
		else:
			return True

	def makeTaskSpec(self):
		ts = TaskSpecRLGlue.TaskSpec(discount_factor=1.0, reward_range=(-1.0, 0.0))
		ts.addDiscreteAction((0, 3))
		ts.addContinuousObservation((0.0, self.size[0]))
		ts.addContinuousObservation((0.0, self.size[1]))
		ts.setEpisodic()
		ts.setExtra("Continuous MultiRoom Gridworld by Will Dabney")
		return ts.toTaskSpec()

	def env_init(self):
		return self.makeTaskSpec()
	
	def env_start(self):
		if self.random_start:
			self.pos = self.fixPoint(numpy.random.random((2,)) * self.size)
		else:
			self.pos[:] = 0.0

		returnObs = Observation()
		returnObs.doubleArray=self.pos.tolist()
		return returnObs
		
	def isAtGoal(self):
		return numpy.linalg.norm(self.pos - self.goal) < self.fudge

	def env_step(self,thisAction):
		episodeOver = 0
		theReward = -1.0
		intAction = thisAction.intArray[0]

		if intAction == 0:
			self.pos[0] += 1.0
		elif intAction == 1:
			self.pos[0] -= 1.0
		elif intAction == 2:
			self.pos[1] += 1.0
		elif intAction == 3:
			self.pos[1] -= 1.0
		
		if self.noise > 0:
			self.pos += numpy.random.normal(scale=self.noise, size=(2,))

		#self.pos = self.pos.clip([0, 0], self.size)
		self.pos = self.fixPoint(self.pos)

		if self.isAtGoal():
			theReward = 0.0
			episodeOver = 1

		theObs = Observation()
		theObs.doubleArray = self.pos.tolist()
		
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
	parser = argparse.ArgumentParser(description='Run 2D MultiRoom Noisy Continuous Gridworld environment in network mode.')
	parser.add_argument("--size_x", type=float, default=10, help="Size of the gridworld in the x (horizontal) dimension, where 1.0 is the unit of movement.")
	parser.add_argument("--size_y", type=float, default=10, help="Size of the gridworld in the y (vertical) dimension, where 1.0 is the unit of movement.")
	parser.add_argument("--goal_x", type=float, default=10, help="Goal x coordinate")
	parser.add_argument("--goal_y", type=float, default=10, help="Goal y coordinate")
	parser.add_argument("--noise", type=float, default=0, help="Standard deviation of additive noise to generate")
	parser.add_argument("--fudge", type=float, default=1.4143, help="Distance from goal allowed before episode is counted as finished")
	parser.add_argument("--random_restarts", type=bool, default=False, help="Goal y coordinate")
	args = parser.parse_args()
	EnvironmentLoader.loadEnvironment(MultiRoomGridworld(args.size_x, args.size_y, args.goal_x, args.goal_y, noise=args.noise, random_start=args.random_restarts, fudge=args.fudge))
