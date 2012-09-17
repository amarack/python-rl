
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
import fourier

# This is an implementation of the Temporal Difference with Correction (TDC) algorithm, 
# for off-policy reinforcement learning.
# 

class tdc_agent(Agent):

	def __init__(self, filename=None):
		# Experience saver object:
		if filename is not None:
			self.expSaver = expreplay.experience_saver("tdc_saver.pickle")
		else:
			self.expSaver = None

		self.basis = None
		self.randGenerator=Random()
		self.lastAction=Action()
		self.lastObservation=Observation()
		self.sarsa_stepsize = 0.05
		self.meta_stepsize = 0.05
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

			
			self.basis = fourier.FourierBasis(self.numStates, 3, TaskSpec.getDoubleObservations())
			self.value_function=numpy.zeros((self.basis.numTerms, self.numActions))
			self.w_function = numpy.zeros((self.basis.numTerms, self.numActions))

		else:
			print "Task Spec could not be parsed: "+taskSpecString;
			
		self.lastAction=Action()
		self.lastObservation=Observation()
		
	def egreedy(self, state):
		maxIndex=0
		a=1
		if not self.exploringFrozen and self.randGenerator.random()<self.sarsa_epsilon:
			return self.randGenerator.randint(0,self.numActions-1)
                
		return numpy.dot(self.value_function.T, self.basis.computeFeatures(state)).argmax()
	
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

	# Experience-Replay / Off-Policy alternative to agent_start which also specifies the first action taken in an episode
	def offpolicy_start(self,observation, returnAction):
		theState=numpy.array(observation.doubleArray)
		
		self.lastAction=copy.deepcopy(returnAction)
		self.lastObservation=copy.deepcopy(observation)
		
	def agent_step(self,reward, observation):
		newState=numpy.array(observation.doubleArray)
		lastState=numpy.array(self.lastObservation.doubleArray)
		lastAction=self.lastAction.intArray[0]

		newIntAction=self.egreedy(newState)

		Q_sa=numpy.dot(self.value_function[:,lastAction], self.basis.computeFeatures(lastState))
		Q_sprime_aprime=numpy.dot(self.value_function[:,newIntAction], self.basis.computeFeatures(newState))

		delta = (reward + self.sarsa_gamma * Q_sprime_aprime - Q_sa)


		if not self.policyFrozen:
			update = delta * self.basis.computeFeatures(lastState) - \
			    self.sarsa_gamma * self.basis.computeFeatures(newState) * numpy.dot(self.basis.computeFeatures(lastState), self.w_function[:,lastAction])

			self.value_function[:,lastAction] += self.sarsa_stepsize * update

			self.w_function[:,lastAction] += self.meta_stepsize * \
			    (delta - numpy.dot(self.basis.computeFeatures(lastState), self.w_function[:,lastAction])) * self.basis.computeFeatures(lastState)


		returnAction=Action()
		returnAction.intArray=[newIntAction]
		
		self.lastAction=copy.deepcopy(returnAction)
		self.lastObservation=copy.deepcopy(observation)

		if not self.policyFrozen:
			if self.expSaver is not None:
				self.expSaver.addStep(observation, returnAction, reward)

		return returnAction

	# Off-policy alternative to agent_step which also specifies the action taken out of this observation state
	def offpolicy_step(self,observation, action, reward):
		newState=numpy.array(observation.doubleArray)
		lastState=numpy.array(self.lastObservation.doubleArray)
		lastAction=self.lastAction.intArray[0]

		newIntAction=action.intArray[0]

		Q_sa=numpy.dot(self.value_function[:,lastAction], self.basis.computeFeatures(lastState))
		Q_sprime_aprime=numpy.dot(self.value_function[:,newIntAction], self.basis.computeFeatures(newState))

		delta = (reward + self.sarsa_gamma * Q_sprime_aprime - Q_sa)

		lastStateBasis = self.basis.computeFeatures(lastState)
		prod = numpy.zeros((len(lastStateBasis), len(lastStateBasis)))
		for i in range(len(lastStateBasis)):
			for j in range(len(lastStateBasis)):
				prod[i,j] = lastStateBasis[i]*lastStateBasis[j]
		print delta**2 * numpy.dot(lastStateBasis, numpy.dot(numpy.linalg.pinv(prod), lastStateBasis))

		if not self.policyFrozen:
			update = delta * lastStateBasis - \
			    self.sarsa_gamma * self.basis.computeFeatures(newState) * numpy.dot(lastStateBasis, self.w_function[:,lastAction])

			self.value_function[:,lastAction] += self.sarsa_stepsize * update

			self.w_function[:,lastAction] += self.meta_stepsize * \
			    (delta - numpy.dot(lastStateBasis, self.w_function[:,lastAction])) * lastStateBasis

			self.value_function[:,lastAction] += self.sarsa_stepsize * delta * lastStateBasis

		returnAction=Action()
		returnAction.intArray=[newIntAction]
		
		self.lastAction=copy.deepcopy(returnAction)
		self.lastObservation=copy.deepcopy(observation)

		return returnAction

	
	def agent_end(self,reward):
		lastState=self.lastObservation.doubleArray
		lastAction=self.lastAction.intArray[0]

		Q_sa=numpy.dot(self.value_function[:,lastAction], self.basis.computeFeatures(lastState))

		delta = (reward - Q_sa)

		if not self.policyFrozen:
			update = delta * self.basis.computeFeatures(lastState)
			self.value_function[:,lastAction] += self.sarsa_stepsize * update

			self.w_function[:,lastAction] += self.meta_stepsize * \
			    (delta - numpy.dot(self.basis.computeFeatures(lastState), self.w_function[:,lastAction])) * self.basis.computeFeatures(lastState)


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
