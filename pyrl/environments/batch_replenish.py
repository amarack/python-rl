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
class BatchReplenishment(Environment):
	"""Batch replenishment inventory control task.

	The domain was given by George and Powell 2006. It is an example of a simple 
	domain in which no fixed step-size performs well, but adaptive step-sizes 
	do well.
	"""

	name = "Batch Replenishment"

	def __init__(self, demand_mean = 10.0, demand_std = 1.0, payoff = 5., 
		     cost = 2., gamma = 0.99, time_period = 20, noise=0.0):

		self.T = time_period
		self.noise = noise
		self.demand = numpy.array([demand_mean, demand_std])
		self.payoff = payoff
		self.cost = cost
		self.discount = gamma
		self.max_quantity = 200.
		self.domain_name = "Noisy Batch Replenishment Problem"

	def makeTaskSpec(self):
		ts = TaskSpecRLGlue.TaskSpec(discount_factor=self.discount, 
					     reward_range=(-self.max_quantity * self.cost, 
							    self.max_quantity * self.payoff))
		ts.addDiscreteAction((0, 3)) # Representing purchase of 0, 1, 10, and 100 units
		ts.addContinuousObservation((0.0, self.max_quantity))
		ts.addContinuousObservation((0.0, self.max_quantity))
		ts.setEpisodic()
		ts.setExtra(self.domain_name)
		return ts.toTaskSpec()

	def reset(self):
		# Start with no resources in stock, and no unsatisfied demand
		self.state = numpy.zeros((2,))
		self.counter = 0

	def env_init(self):
		return self.makeTaskSpec()
	
	def env_start(self):
		self.reset()
		returnObs = Observation()
		returnObs.doubleArray = self.state.tolist()
		return returnObs
		
	def takeAction(self, intAction):
		x = 0. if intAction == 0 else 10.**(intAction-1)
		self.counter += 1
		# If noisy, create noise on cost/payoff
		paynoise = numpy.random.normal(scale=self.noise) if self.noise > 0 else 0.0
		costnoise = numpy.random.normal(scale=self.noise) if self.noise > 0 else 0.0

		# Update random demand
		self.state[1] = min(self.max_quantity, 
				    max(0., numpy.random.normal(self.demand[0], scale=self.demand[1])))
		reward = (self.payoff + paynoise) * self.state.min() - (self.cost + costnoise) * x
		self.state[0] = min(self.max_quantity, max(0., self.state[0] - self.state[1]) + x)

		
		return reward/600.

	def env_step(self,thisAction):
		intAction = thisAction.intArray[0]
		theReward = self.takeAction(intAction)

		theObs = Observation()
		theObs.doubleArray = self.state.tolist()
		
		returnRO = Reward_observation_terminal()
		returnRO.r = theReward
		returnRO.o = theObs
		returnRO.terminal = int(self.counter >= self.T)

		return returnRO

	def env_cleanup(self):
		pass

	def env_message(self,inMessage):
		return "I don't know how to respond to your message";


if __name__=="__main__":
	import argparse
	parser = argparse.ArgumentParser(description='Run 2D Noisy Continuous Gridworld environment in network mode.')
	parser.add_argument("--demand_mean", type=float, default=10., help="Mean demand for the product.")
	parser.add_argument("--demand_std", type=float, default=1., 
			    help="Standard deviation of demand for the product.")
	parser.add_argument("--payoff", type=float, default=5., help="Payment received per unit product sold.")
	parser.add_argument("--cost", type=float, default=2., help="Cost per unit product purchased.")
	parser.add_argument("--discount_factor", type=float, default=0.999, help="Discount factor to learn over.")
	parser.add_argument("--time_period", type=int, default=20, help="Time period for problem. (Number of steps to run)")
	parser.add_argument("--noise", type=float, default=0, help="Standard deviation of additive noise to generate")
	args = parser.parse_args()
	EnvironmentLoader.loadEnvironment(BatchReplenishment(demand_mean=args.demand_mean, 
							     demand_std=args.demand_std, 
							     payoff=args.payoff, 
							     cost=args.cost, 
							     gamma=args.discount_factor,
							     time_period = args.time_period,
							     noise=args.noise))
