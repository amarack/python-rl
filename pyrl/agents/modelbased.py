
from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.utils import TaskSpecVRLGLUE3

from random import Random
import numpy
import sys
import copy

#import pyrl.basis.fourier as fourier
#import pyrl.basis.rbf as rbf
#import pyrl.basis.tilecode as tilecode

from pyrl.agents.models import batch_model
from pyrl.agents.planners import fitted_qiteration

class ModelBasedAgent(Agent):

	def __init__(self, model, planner, model_params={}, planner_params={}):
		self.randGenerator = Random()	
		self.lastAction=Action()
		self.lastObservation=Observation()
		
		self.model_class = model
		self.planner_class = planner
		self.model = None
		self.planner = None
		self.model_params = model_params
		self.planner_params = planner_params

	def agent_init(self,taskSpec):
		# Parse the task specification and set up the weights and such
		TaskSpec = TaskSpecVRLGLUE3.TaskSpecParser(taskSpec)
		if TaskSpec.valid:
			# Check observation form, and then set up number of features/states
			assert len(TaskSpec.getDoubleObservations()) + len(TaskSpec.getIntObservations()) >0, "expecting at least one continuous or discrete observation"
			self.numStates=len(TaskSpec.getDoubleObservations())
			self.discStates = numpy.array(TaskSpec.getIntObservations())
			self.numDiscStates = int(reduce(lambda a, b: a * (b[1] - b[0] + 1), self.discStates, 1.0)) #if len(self.discStates) > 0 else 0

			# Check action form, and then set number of actions
			assert len(TaskSpec.getIntActions())==1, "expecting 1-dimensional discrete actions"
			assert len(TaskSpec.getDoubleActions())==0, "expecting no continuous actions"
			assert not TaskSpec.isSpecial(TaskSpec.getIntActions()[0][0]), " expecting min action to be a number not a special value"
			assert not TaskSpec.isSpecial(TaskSpec.getIntActions()[0][1]), " expecting max action to be a number not a special value"
			self.numActions=TaskSpec.getIntActions()[0][1]+1;
			
			self.model = self.model_class(self.numDiscStates, TaskSpec.getDoubleObservations(), self.numActions, TaskSpec.getRewardRange()[0], self.model_params)
			self.planner = self.planner_class(self.model, self.planner_params)
			
		else:
			print "Task Spec could not be parsed: "+taskSpecString;

		self.lastAction=Action()
		self.lastObservation=Observation()


	def getAction(self, state, discState):
		s = numpy.zeros((len(state) + 1,))
		s[0] = discState
		s[1:] = state
		a = self.planner.getAction(s)
		return a
		
	def getDiscState(self, state):
		if self.numDiscStates > 1:
			x = numpy.zeros((self.numDiscStates,))
			mxs = self.discStates[:,1] - self.discStates[:,0] + 1
			mxs = numpy.array(list(mxs[:0:-1].cumprod()[::-1]) + [1])
			x = numpy.array(state) - self.discStates[:,0]
			return (x * mxs).sum()
		else:
			return 0

	def agent_start(self,observation):
		theState = numpy.array(list(observation.doubleArray))
		thisIntAction=self.getAction(theState, self.getDiscState(observation.intArray))
		returnAction=Action()
		returnAction.intArray=[thisIntAction]

		self.lastAction=copy.deepcopy(returnAction)
		self.lastObservation=copy.deepcopy(observation)
		
		return returnAction
	
	def agent_step(self,reward, observation):
		newState = numpy.array(list(observation.doubleArray))
		lastState = numpy.array(list(self.lastObservation.doubleArray))
		lastAction = self.lastAction.intArray[0]

		newDiscState = self.getDiscState(observation.intArray)
		lastDiscState = self.getDiscState(self.lastObservation.intArray)

		phi_t = numpy.zeros((self.numStates+1,))
		phi_tp = numpy.zeros((self.numStates+1,))
		phi_t[0] = lastDiscState
		phi_t[1:] = lastState
		phi_tp[0] = newDiscState
		phi_tp[1:] = newState

		self.planner.updateExperience(phi_t, lastAction, phi_tp, reward)

		newIntAction = self.getAction(newState, newDiscState)
		returnAction=Action()
		returnAction.intArray=[newIntAction]
		
		self.lastAction=copy.deepcopy(returnAction)
		self.lastObservation=copy.deepcopy(observation)
		return returnAction

	def agent_end(self,reward):
		lastState = numpy.array(list(self.lastObservation.doubleArray))
		lastAction = self.lastAction.intArray[0]
		lastDiscState = self.getDiscState(self.lastObservation.intArray)

		phi_t = numpy.zeros((self.numStates+1,))
		phi_t[0] = lastDiscState
		phi_t[1:] = lastState
		self.planner.updateExperience(phi_t, lastAction, None, reward)

	def agent_cleanup(self):
		pass
	
	def agent_message(self,inMessage):
		return "ModelBasedAgent(Python) does not understand your message."


if __name__=="__main__":
	import argparse
	parser = argparse.ArgumentParser(description='Run ModelBasedAgent in network mode')
	parser.add_argument("--model", type=float, default=0.1, help="What model to use... not filled out yet")
	args = parser.parse_args()
	model_params = {}
	planner_params = {}
	AgentLoader.loadAgent(ModelBasedAgent(batch_model.BatchModel, fitted_qiteration.FittedQIteration, model_params, planner_params))
