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

@register_environment
class MultiRoomGridworld(gridworld.Gridworld):
	name = "Multi-Room Gridworld"

	# All parameters are in units of 1, where 1 is how far on average
	# the agent can move with a single action.
	# The walls will always be of unit thickness and be placed 
	# at 0.5*size_y with a door at 0.9*size_x, and 
	# above that wall a vertical wall will be placed at 0.3*size_x with a door at 0.75*size_y
	# If the goal falls inside a wall it will be pushed to the nearest non-wall location
	def __init__(self, size_x=10, size_y=10, goal_x=10, goal_y=10, noise=0.0, random_start=False, fudge=1.4143):
		gridworld.Gridworld.__init__(self, size_x=size_x, size_y=size_y, goal_x=goal_x, goal_y=goal_y, 
					     noise=noise, random_start=random_start, fudge=fudge)
		# Build walls and doors (actually might only need to specify the doors)
		#self.wall1 = numpy.array([[0.0, size_y*0.5], [size_x, size_y*0.5]])
		self.door1 = numpy.array([size_x*0.9, size_y*0.5])
		#self.wall2 = numpy.array([[size_x*0.3, size_y*0.5], [size_x*0.3, size_y]])
		self.door2 = numpy.array([size_x*0.3, size_y*0.75])
		self.goal = self.fixPoint(self.goal)
		self.domain_name = "Continuous MultiRoom Gridworld by Will Dabney"

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

	def reset(self):
		if self.random_start:
			self.pos = self.fixPoint(numpy.random.random((2,)) * self.size)
		else:
			self.pos[:] = 0.0
	
	def takeAction(self, action):
		reward = gridworld.Gridworld.takeAction(self, action)
		self.pos = self.fixPoint(self.pos)
		return reward


if __name__=="__main__":
	import argparse
	parser = argparse.ArgumentParser(description='Run 2D MultiRoom Noisy Continuous Gridworld environment in network mode.')
	gridworld.addGridworldArgs(parser)
	args = parser.parse_args()
	EnvironmentLoader.loadEnvironment(MultiRoomGridworld(size_x=args.size_x, size_y=args.size_y, goal_x=args.goal_x, goal_y=args.goal_y, noise=args.noise, random_start=args.random_restarts, fudge=args.fudge))
