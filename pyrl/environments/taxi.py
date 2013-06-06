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
import random

from rlglue.environment.Environment import Environment
from rlglue.environment import EnvironmentLoader as EnvironmentLoader
from rlglue.types import Observation
from rlglue.types import Action
from rlglue.types import Reward_observation_terminal
from pyrl.rlglue import TaskSpecRLGlue
from pyrl.rlglue.registry import register_environment

@register_environment
class Taxi(Environment):
	name = "Taxi"

	# All parameters are in units of 1, where 1 is how far on average
	# the agent can move with a single action.
	def __init__(self, size_x=5, size_y=5, walls=None, landmarks=None, fuel_loc=numpy.array([2.0, 1.0]), fickleness=0.0, noise=0.0, fudge=1.4143):
		self.size = numpy.array([size_x, size_y])
		if landmarks is None: # Use original Taxi landmarks
			self.landmarks = numpy.array([[0.0, 0.0], [0.0, 4.0], [3.0, 0.0], [4.0, 4.0]])
		else:
			self.landmarks = numpy.array(landmarks)
			if len(self.landmarks) < 2:
				print "Must include at least two landmarks"
				sys.exit(1)

		# Walls are specified by giving the x-coor. where they are located and
		# how far up from the bottom (y > 0) or down from the top (y < 0) they extend.
		if walls is None:
			self.walls = numpy.array([[1.0, 2.0], [2.0, -2.0], [3.0, 2.0]])
		else:
			self.walls = numpy.array(walls)

		self.fuel = 0
		self.fuel_loc = fuel_loc
		self.pass_loc = 0 # Passenger location: -1 for in taxi, >= 0 for a landmark
		self.pass_dest = 0 # Passenger destination: >=0 for a landmark
		self.fickleness = fickleness # Probability of passenger changing their mind mid trip
		self.noise = noise
		self.pos = numpy.zeros((2,))
		self.fudge = fudge
		self.domain_name = "Continuous Taxi Domain by Will Dabney"

	def makeTaskSpec(self):
		ts = TaskSpecRLGlue.TaskSpec(discount_factor=1.0, reward_range=(-20.0, 20.0))
		if self.fuel_loc is not None:
			ts.addDiscreteAction((0, 6)) # N,S,E,W,Pickup,Dropoff,Refuel
		else:
			ts.addDiscreteAction((0, 5)) # N,S,E,W,Pickup,Dropoff
		ts.addContinuousObservation((0.0, self.size[0]-1)) # x
		ts.addContinuousObservation((0.0, self.size[1]-1)) # y
		if self.fuel_loc is not None:
			ts.addContinuousObservation((-1.0, 12.0)) # Fuel level
		ts.addDiscreteObservation((-1, len(self.landmarks)-1)) # Passenger location
		ts.addDiscreteObservation((0, len(self.landmarks)-1)) # Passenger destination
		ts.setEpisodic()
		ts.setExtra(self.domain_name)
		return ts.toTaskSpec()

	def reset(self):
		self.pos = numpy.random.random((2,)) * self.size
		self.fuel = numpy.random.random()*7 + 5.0
		self.lm_list = range(len(self.landmarks))
		random.shuffle(self.lm_list)
		self.pass_loc = self.lm_list.pop()
		self.pass_dest = random.choice(self.lm_list)

	def makeObservation(self):
		returnObs = Observation()
		returnObs.doubleArray = self.pos.tolist()
		if self.fuel_loc is not None:
			returnObs.doubleArray += [self.fuel]
		returnObs.intArray = [self.pass_loc, self.pass_dest]
		return returnObs

	def env_init(self):
		return self.makeTaskSpec()

	def env_start(self):
		self.reset()
		return self.makeObservation()


	def atPoint(self, point):
		return numpy.linalg.norm(self.pos -point) < self.fudge

	def isAtGoal(self):
		return self.pass_loc == self.pass_dest

	def takeAction(self, intAction):
		reward = -1.0
		self.fuel -= 1
		prev_pos = self.pos.copy()
		sign = 0
		if intAction == 0:
			self.pos[0] += 1.0
			sign = 1
		elif intAction == 1:
			self.pos[0] -= 1.0
			sign = -1
		elif intAction == 2:
			self.pos[1] += 1.0
		elif intAction == 3:
			self.pos[1] -= 1.0
		elif intAction == 4: # Pickup
			if self.pass_loc >= 0 and self.atPoint(self.landmarks[self.pass_loc]):
				self.pass_loc = -1
			else:
				reward = -10.0
		elif intAction == 5: # Drop off
			if self.pass_loc == -1 and self.atPoint(self.landmarks[self.pass_dest]):
				self.pass_loc = self.pass_dest
				reward = 20.0
			else:
				reward = -10.0
		elif self.fuel_loc is not None and intAction == 4: # Refuel
			if self.atPoint(self.fuel_loc):
				self.fuel = 12.0

		if self.noise > 0:
			self.pos += numpy.random.normal(scale=self.noise, size=(2,))

		self.pos = self.pos.clip([0, 0], self.size)

		if sign != 0 and self.hitsWall(prev_pos, self.pos, sign):
			self.pos[0] = prev_pos[0] # Only revert the x-coord, to allow noise and such in y

		if numpy.random.random() < self.fickleness:
			self.pass_dest = random.choice(self.lm_list)

		return reward

	def hitsWall(self, old_pos, new_pos, sign):
		return (((self.walls[:,0]*sign >= old_pos[0]*sign) & (self.walls[:,0]*sign < new_pos[0]*sign)) \
				& ((self.walls[:,1] > old_pos[1]) | ((self.size[1]-1)+self.walls[:,1] < old_pos[1]))).any()

	def env_step(self,thisAction):
		episodeOver = 0
		theReward = -1.0
		intAction = thisAction.intArray[0]

		theReward = self.takeAction(intAction)

		if self.isAtGoal() or (self.fuel_loc is not None and self.fuel) < 0:
			episodeOver = 1

		theObs = self.makeObservation()
		returnRO = Reward_observation_terminal()
		returnRO.r = theReward
		returnRO.o = theObs
		returnRO.terminal = episodeOver

		return returnRO

	def env_cleanup(self):
		pass

	def env_message(self,inMessage):
		return "I don't know how to respond to your message";

def addTaxiArgs(parser):
	parser.add_argument("--size_x", type=float, default=5, help="Size of the gridworld in the x (horizontal) dimension, where 1.0 is the unit of movement.")
	parser.add_argument("--size_y", type=float, default=5, help="Size of the gridworld in the y (vertical) dimension, where 1.0 is the unit of movement.")
	parser.add_argument("--landmark", action="append", nargs=2, help="Add a landmark, give x y coordinates", type=float)
	parser.add_argument("--wall", type=float, action="append", nargs=2, help="Add a wall, give x coordinate and size in y with sign indicating starting at the bottom (+) or top (-)")
	parser.add_argument("--fuel_loc", type=float, default=[2.0, 1.0], nargs=2, help="x y coordinate of the fuel station")
	parser.add_argument("--fickleness", type=float, default=0, help="Probability of the passenger changing their destination mid-route.")
	parser.add_argument("--noise", type=float, default=0, help="Standard deviation of additive noise to generate")
	parser.add_argument("--fudge", type=float, default=1.4143, help="Distance from goal allowed before episode is counted as finished")


if __name__=="__main__":
	import argparse
	parser = argparse.ArgumentParser(description='Run 2D Noisy Continuous Gridworld environment in network mode.')
	addTaxiArgs(parser)
	args = parser.parse_args()
	fuelloc = None if args.fuel_loc[0] < 0 else args.fuel_loc
	walls = numpy.array(args.wall) if args.wall is not None else None
	landmarks = numpy.array(args.landmark) if args.landmark is not None else None
	EnvironmentLoader.loadEnvironment(Taxi(args.size_x, args.size_y, walls=walls, landmarks=landmarks, fuel_loc=fuelloc, fickleness=args.fickleness, noise=args.noise, fudge=args.fudge))

