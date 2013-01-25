# 
# Copyright (C) 2008, Brian Tanner
# 
#http://rl-glue-ext.googlecode.com/
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

import random
import sys
import copy
from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation

from random import Random


#ds = UnsupervisedDataSet(6)
#ds.addSample([0, 1] * 3)
#ds.addSample([1, 0] * 3)

#cfg = RbmGibbsTrainerConfig()
#cfg.maxIter = 3

#rbm = Rbm.fromDims(6, 1)
#trainer = RbmBernoulliTrainer(rbm, ds, cfg)
#print rbm.params, rbm.biasParams
#for _ in xrange(50):
#    trainer.train()


class skeleton_agent(Agent):
	randGenerator=Random()
	lastAction=Action()
	lastObservation=Observation()
	
	def agent_init(self,taskSpec):
		#See the sample_sarsa_agent in the mines-sarsa-example project for how to parse the task spec
		self.lastAction=Action()
		self.lastObservation=Observation()
		
	def agent_start(self,observation):
		#Generate random action, 0 or 1
		thisIntAction=self.randGenerator.randint(0,1)
		returnAction=Action()
		returnAction.intArray=[thisIntAction]
		
		lastAction=copy.deepcopy(returnAction)
		lastObservation=copy.deepcopy(observation)

		return returnAction
	
	def agent_step(self,reward, observation):
		#Generate random action, 0 or 1
		thisIntAction=self.randGenerator.randint(0,1)
		returnAction=Action()
		returnAction.intArray=[thisIntAction]
		
		lastAction=copy.deepcopy(returnAction)
		lastObservation=copy.deepcopy(observation)

		return returnAction
	
	def agent_end(self,reward):
		pass
	
	def agent_cleanup(self):
		pass
	
	def agent_message(self,inMessage):
		if inMessage=="what is your name?":
			return "my name is skeleton_agent, Python edition!";
		else:
			return "I don't know how to respond to your message";


if __name__=="__main__":
	AgentLoader.loadAgent(skeleton_agent())
