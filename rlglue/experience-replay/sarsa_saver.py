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
#  $Revision: 1011 $
#  $Date: 2009-02-11 22:29:54 -0700 (Wed, 11 Feb 2009) $
#  $Author: brian@tannerpages.com $
#  $HeadURL: http://rl-library.googlecode.com/svn/trunk/projects/packages/examples/mines-sarsa-python/sample_sarsa_agent.py $

import random
import sys
import copy
import pickle
import numpy

from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.utils import TaskSpecVRLGLUE3
from random import Random
import expreplay


# This is a very simple Sarsa agent for discrete-action, discrete-state
# environments.  It uses epsilon-greedy exploration.
# 
# We've made a decision to store the previous action and observation in 
# their raw form, as structures.  This code could be simplified and you
# could store them just as ints.


# TO USE THIS Agent [order doesn't matter]
# NOTE: I'm assuming the Python codec is installed an is in your Python path
#   -  Start the rl_glue executable socket server on your computer
#   -  Run the SampleMinesEnvironment and SampleExperiment from this or a
#   different codec (Matlab, Python, Java, C, Lisp should all be fine)
#   -  Start this agent like:
#   $> python sample_sarsa_agent.py

class sarsa_agent(Agent):

	def __init__(self, filename=None):
		# Experience saver object:
		if filename is not None:
			self.expSaver = expreplay.experience_saver("sarsa_saver.pickle")
		else:
			self.expSaver = None

		self.randGenerator=Random()
		self.lastAction=Action()
		self.lastObservation=Observation()
		self.sarsa_stepsize = 0.1
		self.sarsa_epsilon = 0.1
		self.sarsa_gamma = 1.0
		self.numStates = 0
		self.numActions = 0
		self.value_function = None
	
		self.policyFrozen=False
		self.exploringFrozen=False
	
	def agent_init(self,taskSpecString):
		if self.expSaver is not None:
			self.expSaver.setTaskSpecString(taskSpecString)

		TaskSpec = TaskSpecVRLGLUE3.TaskSpecParser(taskSpecString)
		if TaskSpec.valid:
			assert len(TaskSpec.getDoubleObservations())>0, "expecting at least one continuous observation"
			self.numStates=len(TaskSpec.getDoubleObservations())

			assert len(TaskSpec.getIntActions())==1, "expecting 1-dimensional discrete actions"
			assert len(TaskSpec.getDoubleActions())==0, "expecting no continuous actions"
			assert not TaskSpec.isSpecial(TaskSpec.getIntActions()[0][0]), " expecting min action to be a number not a special value"
			assert not TaskSpec.isSpecial(TaskSpec.getIntActions()[0][1]), " expecting max action to be a number not a special value"
			self.numActions=TaskSpec.getIntActions()[0][1]+1;
			
			self.value_function=numpy.zeros((self.numStates, self.numActions))

		else:
			print "Task Spec could not be parsed: "+taskSpecString;
			
		self.lastAction=Action()
		self.lastObservation=Observation()
		
	def egreedy(self, state):
		maxIndex=0
		a=1
		if not self.exploringFrozen and self.randGenerator.random()<self.sarsa_epsilon:
			return self.randGenerator.randint(0,self.numActions-1)
                
		return numpy.dot(self.value_function.T, state).argmax()
	
	def agent_start(self,observation):
		theState=numpy.array(observation.doubleArray)
		thisIntAction=self.egreedy(theState)
		returnAction=Action()
		returnAction.intArray=[thisIntAction]
		
		self.lastAction=copy.deepcopy(returnAction)
		self.lastObservation=copy.deepcopy(observation)
		
		if not self.policyFrozen:
			if self.expSaver is not None:
				self.expSaver.startEpisode(observation, returnAction)
		return returnAction

	def offpolicy_start(self,observation, returnAction):
		theState=numpy.array(observation.doubleArray)
		
		self.lastAction=copy.deepcopy(returnAction)
		self.lastObservation=copy.deepcopy(observation)
		
	def agent_step(self,reward, observation):
		newState=numpy.array(observation.doubleArray)
		lastState=numpy.array(self.lastObservation.doubleArray)
		lastAction=self.lastAction.intArray[0]

		newIntAction=self.egreedy(newState)

		Q_sa=numpy.dot(self.value_function[:,lastAction], lastState)
		Q_sprime_aprime=numpy.dot(self.value_function[:,newIntAction], newState)

		delta = (reward + self.sarsa_gamma * Q_sprime_aprime - Q_sa)


		if not self.policyFrozen:
			self.value_function[:,lastAction] += self.sarsa_stepsize * delta * lastState

		returnAction=Action()
		returnAction.intArray=[newIntAction]
		
		self.lastAction=copy.deepcopy(returnAction)
		self.lastObservation=copy.deepcopy(observation)

		if not self.policyFrozen:
			if self.expSaver is not None:
				self.expSaver.addStep(observation, returnAction, reward)

		return returnAction

	def offpolicy_step(self,observation, action, reward):
		newState=numpy.array(observation.doubleArray)
		lastState=numpy.array(self.lastObservation.doubleArray)
		lastAction=self.lastAction.intArray[0]

		newIntAction=action.intArray[0]

		Q_sa=numpy.dot(self.value_function[:,lastAction], lastState)
		Q_sprime_aprime=numpy.dot(self.value_function[:,newIntAction], newState)

		delta = (reward + self.sarsa_gamma * Q_sprime_aprime - Q_sa)

		print delta
		if not self.policyFrozen:
			self.value_function[:,lastAction] += self.sarsa_stepsize * delta * lastState

		returnAction=Action()
		returnAction.intArray=[newIntAction]
		
		self.lastAction=copy.deepcopy(returnAction)
		self.lastObservation=copy.deepcopy(observation)

		return returnAction

	
	def agent_end(self,reward):
		lastState=self.lastObservation.doubleArray
		lastAction=self.lastAction.intArray[0]

		Q_sa=numpy.dot(self.value_function[:,lastAction], lastState)

		delta = (reward - Q_sa)

		if not self.policyFrozen:
			self.value_function[:,lastAction] += self.sarsa_stepsize * delta * lastState
			if self.expSaver is not None:
				self.expSaver.endEpisode(reward)
		
	def agent_cleanup(self):
		if self.expSaver is not None:
			self.expSaver.flush()

	def save_value_function(self, fileName):
		theFile = open(fileName, "w")
		pickle.dump(self.value_function, theFile)
		theFile.close()

	def load_value_function(self, fileName):
		theFile = open(fileName, "r")
		self.value_function=pickle.load(theFile)
		theFile.close()
	
	def agent_message(self,inMessage):
		
		#	Message Description
	 	# 'freeze learning'
		# Action: Set flag to stop updating policy
		#
		if inMessage.startswith("freeze learning"):
			self.policyFrozen=True
			return "message understood, policy frozen"

		#	Message Description
	 	# unfreeze learning
	 	# Action: Set flag to resume updating policy
		#
		if inMessage.startswith("unfreeze learning"):
			self.policyFrozen=False
			return "message understood, policy unfrozen"

		#Message Description
	 	# freeze exploring
	 	# Action: Set flag to stop exploring (greedy actions only)
		#
		if inMessage.startswith("freeze exploring"):
			self.exploringFrozen=True
			return "message understood, exploring frozen"

		#Message Description
	 	# unfreeze exploring
	 	# Action: Set flag to resume exploring (e-greedy actions)
		#
		if inMessage.startswith("unfreeze exploring"):
			self.exploringFrozen=False
			return "message understood, exploring frozen"

		#Message Description
	 	# save_policy FILENAME
	 	# Action: Save current value function in binary format to 
		# file called FILENAME
		#
		if inMessage.startswith("save_policy"):
			splitString=inMessage.split(" ");
			self.save_value_function(splitString[1]);
			print "Saved.";
			return "message understood, saving policy"

		#Message Description
	 	# load_policy FILENAME
	 	# Action: Load value function in binary format from 
		# file called FILENAME
		#
		if inMessage.startswith("load_policy"):
			splitString=inMessage.split(" ")
			self.load_value_function(splitString[1])
			print "Loaded."
			return "message understood, loading policy"

		return "SampleSarsaAgent(Python) does not understand your message."



if __name__=="__main__":
	AgentLoader.loadAgent(sarsa_agent(filename="sarsa_saver.pickle"))
