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
class WindyGridworld(gridworld.Gridworld):
	name = "Windy Gridworld"
	# The effect of the wind is always positive in the y dimension, and 
	# is equal to the wind_power multiplied with the pdf of the current x-coordinate on a Gaussian distribution 
	# with mean wind_center and standard deviation wind_stdev.
	def __init__(self, size_x=10, size_y=10, goal_x=10, goal_y=10, wind_center=7., wind_stdev=1.0, wind_power=2.0, noise=0.0, random_start=False, fudge=1.4143):
		gridworld.Gridworld.__init__(self, size_x=size_x, size_y=size_y, goal_x=goal_x, goal_y=goal_y, noise=noise, random_start=random_start, fudge=fudge)
		self.wind_center = wind_center
		self.wind_stdev = wind_stdev
		self.wind_power = wind_power
		self.domain_name = "Continuous Windy Gridworld by Will Dabney"
		
	def reset(self):
		if self.random_start:
			self.pos = numpy.random.random((2,)) * self.size
		else:
			self.pos = numpy.array([0.0, self.size[1]*0.5])
	
	def takeAction(self, action):
		self.pos[1] += norm.pdf(self.pos[0], self.wind_center, self.wind_stdev) * self.wind_power
		return gridworld.Gridworld.takeAction(self, action)


if __name__=="__main__":
	import argparse
	parser = argparse.ArgumentParser(description='Run 2D MultiRoom Noisy Continuous Gridworld environment in network mode.')
	gridworld.addGridworldArgs(parser)
	parser.add_argument("--wind_center", type=float, default=7, help="Center, or strongest point, in the x-direction of the wind")
	parser.add_argument("--wind_scale", type=float, default=1.0, help="Scale, or width, of the wind effects around the center.")
	parser.add_argument("--wind_power", type=float, default=2.0, help="The power, or strength, of the wind.")
	args = parser.parse_args()
	EnvironmentLoader.loadEnvironment(
		WindyGridworld(args.size_x, args.size_y, args.goal_x, args.goal_y, wind_center=args.wind_center, 
			       wind_stdev=args.wind_scale, wind_power=args.wind_power, noise=args.noise, 
			       random_start=args.random_restarts, fudge=args.fudge))
