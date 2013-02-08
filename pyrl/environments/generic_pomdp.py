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
from scipy import sparse

from rlglue.environment.Environment import Environment
from rlglue.environment import EnvironmentLoader as EnvironmentLoader
from rlglue.types import Observation
from rlglue.types import Action
from rlglue.types import Reward_observation_terminal
#from rlglue.utils import TaskSpecVRLGLUE3
from pyrl.rlglue import TaskSpecRLGlue
from libPOMDP import libpomdp

class POMDPEnvironment(Environment):

	# All parameters are in units of 1, where 1 is how far on average
	# the agent can move with a single action.
	def __init__(self, spec_filename):
		if not libpomdp.readMDP(spec_filename):
			print "ERROR: Unable to load POMDP spec file " + spec_filename
			sys.exit(1)
		self.initial_belief = libpomdp.getInitialBelief()
		self.state = None
		self.O = map(lambda k: self.buildSparseMatrix(k, (libpomdp.getNumStates(),libpomdp.getNumObservations())), libpomdp.getSparseObsMatrix())
		self.P = map(lambda k: self.buildSparseMatrix(k, (libpomdp.getNumStates(), libpomdp.getNumStates())), libpomdp.getSparseTransitionMatrix())
		self.domain_name = "POMDP Environment from " + spec_filename + " by Will Dabney"

	def buildSparseMatrix(self, rcd, shape): # row col data
		return sparse.csr_matrix((rcd[2], (rcd[0],rcd[1])), shape=shape).todense()

	def makeTaskSpec(self):
		ts = TaskSpecRLGlue.TaskSpec(discount_factor=libpomdp.getDiscount(), reward_range=libpomdp.getRewardRange())
		ts.addDiscreteAction((0.0, libpomdp.getNumActions()-1))
		ts.addDiscreteObservation((0.0, libpomdp.getNumObservations()))
		ts.setContinuing() # The POMDP spec is not set up to have episodes
		ts.setExtra(self.domain_name)
		return ts.toTaskSpec()

	def reset(self):
		# Sample state from initial_belief
		self.state = numpy.where(self.initial_belief.cumsum() > numpy.random.random())[0][0]

	def sampleObservation(self, action):
		return numpy.where(numpy.array(self.O[action][self.state].tolist()).cumsum() > numpy.random.random())[0][0]

	def env_init(self):
		return self.makeTaskSpec()
	
	def env_start(self):
		self.reset()
		returnObs = Observation()
		returnObs.doubleArray = [self.sampleObservation(0)]
		return returnObs
		
	def takeAction(self, intAction):
		prev_state = self.state
		self.state = numpy.where(numpy.array(self.P[intAction][self.state].tolist()).cumsum() > numpy.random.random())[0][0]
		obs = self.sampleObservation(intAction)
		reward = libpomdp.getReward(prev_state, intAction, self.state, obs)
		return obs, reward

	def env_step(self,thisAction):
		intAction = thisAction.intArray[0]
		obs, reward = self.takeAction(intAction)

		theObs = Observation()
		theObs.doubleArray = [obs]
		
		returnRO = Reward_observation_terminal()
		returnRO.r = reward
		returnRO.o = theObs
		returnRO.terminal = 0

		return returnRO

	def env_cleanup(self):
		pass

	def env_message(self,inMessage):
		return "I don't know how to respond to your message";

if __name__=="__main__":
	import argparse
	parser = argparse.ArgumentParser(description='Run a specified POMDP in RL-Glue in network mode.')
	parser.add_argument("--pomdp_file", type=str, help="Filename for POMDP spec file to load and use.", required=True)
	args = parser.parse_args()
	EnvironmentLoader.loadEnvironment(POMDPEnvironment(args.pomdp_file))
