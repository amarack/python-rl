
from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.utils import TaskSpecVRLGLUE3

from random import Random
import numpy
import sys
import copy

import pyrl.basis.fourier as fourier
import pyrl.basis.rbf as rbf
import pyrl.basis.tilecode as tilecode
import stepsizes

class sarsa_lambda(Agent):

	def __init__(self, epsilon, alpha, gamma, lmbda, params={}, softmax=False):
		self.randGenerator = Random()	
		self.lastAction=Action()
		self.lastObservation=Observation()

		self.epsilon = epsilon
		self.lmbda = lmbda
		self.gamma = gamma
		self.basis = None
		self.params = params
		self.softmax = softmax

		# Set up the step-size
		self.alpha = float(alpha)

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

			fa_name = self.params.setdefault('name', 'trivial')
			if self.numStates == 0:
				# Only discrete states
				self.numStates = 1
				if fa_name != "trivial":
					print "Error:", fa_name, " basis requires at least one continuous feature. Forcing trivial basis."
					fa_name = "trivial"

			# Set up the function approximation
			if fa_name == 'fourier':
				# number of dimensions, order, and then the dimension ranges
				self.basis = fourier.FourierBasis(self.numStates, 
								  self.params.setdefault('order', 3), 
								  TaskSpec.getDoubleObservations())
				self.weights = numpy.zeros((self.numDiscStates, self.basis.numTerms, self.numActions))
			elif fa_name == 'rbf':
				self.basis = rbf.RBFBasis(self.numStates,
							  self.params.setdefault('num_functions', self.numStates), 
							  self.params.setdefault('beta', 1.0), 
							  TaskSpec.getDoubleObservations())
				self.weights = numpy.zeros((self.numDiscStates, self.params.setdefault('num_functions', self.numStates), self.numActions))
			elif fa_name == 'tile':
				self.basis = tilecode.TileCodingBasis(self.params.setdefault('num_tiles', 100), 
								      self.params.setdefault('num_weights', 2048))
				self.weights = numpy.zeros((self.numDiscStates, self.params.setdefault('num_weights'), self.numActions))
			else:
				self.basis = None
				self.weights = numpy.zeros((self.numDiscStates, self.numStates, self.numActions))

			self.traces = numpy.zeros(self.weights.shape)
			self.init_stepsize(self.weights.shape, self.params)

		else:
			print "Task Spec could not be parsed: "+taskSpecString;

		self.lastAction=Action()
		self.lastObservation=Observation()


	def getAction(self, state, discState):
		"""Get the action under the current policy for the given state.
		
		Args:
			state: The array of continuous state features
			discState: The integer representing the current discrete state value

		Returns:
			The current policy action, or a random action with some probability.
		"""

		if self.softmax:
			return self.sample_softmax(state, discState)
		else:
			return self.egreedy(state, discState)

	def sample_softmax(self, state, discState):
		Q = None
		if self.basis is None:			
			Q = numpy.dot(self.weights[discState,:,:].T, state)
		else:
			Q = numpy.dot(self.weights[discState,:,:].T, self.basis.computeFeatures(state))
		Q = numpy.exp(numpy.clip(Q/self.epsilon, -500, 500)) 
		Q /= Q.sum()
		
		# Would like to use numpy, but haven't upgraded enough (need 1.7)
		# numpy.random.choice(self.numActions, 1, p=Q)
		Q = Q.cumsum()
		return numpy.where(Q >= numpy.random.random())[0][0]
		
	def egreedy(self, state, discState):
		if self.randGenerator.random() < self.epsilon:
			return self.randGenerator.randint(0,self.numActions-1)

		if self.basis is None:
			return numpy.dot(self.weights[discState,:,:].T, state).argmax()
		else:
			return numpy.dot(self.weights[discState,:,:].T, self.basis.computeFeatures(state)).argmax()
		
	def getDiscState(self, state):
		"""Return the integer value representing the current discrete state.
		
		Args:
			state: The array of integer state features

		Returns:
			The integer value representing the current discrete state
		"""

		if self.numDiscStates > 1:
			x = numpy.zeros((self.numDiscStates,))
			mxs = self.discStates[:,1] - self.discStates[:,0] + 1
			mxs = numpy.array(list(mxs[:0:-1].cumprod()[::-1]) + [1])
			x = numpy.array(state) - self.discStates[:,0]
			return (x * mxs).sum()
		else:
			return 0

	def agent_start(self,observation):
		"""Start an episode for the RL agent.

		Args:
			observation: The first observation of the episode. Should be an RLGlue Observation object.

		Returns:
			The first action the RL agent chooses to take, represented as an RLGlue Action object.
		"""

		theState = numpy.array(list(observation.doubleArray))
		thisIntAction=self.getAction(theState, self.getDiscState(observation.intArray))
		returnAction=Action()
		returnAction.intArray=[thisIntAction]

		# Clear traces
		self.traces.fill(0.0)

		self.lastAction=copy.deepcopy(returnAction)
		self.lastObservation=copy.deepcopy(observation)
		
		return returnAction
	
	def agent_step(self,reward, observation):
		"""Take one step in an episode for the agent, as the result of taking the last action.
		
		Args:
			reward: The reward received for taking the last action from the previous state.
			observation: The next observation of the episode, which is the consequence of taking the previous action.

		Returns:
			The next action the RL agent chooses to take, represented as an RLGlue Action object.
		"""

		newState = numpy.array(list(observation.doubleArray))
		lastState = numpy.array(list(self.lastObservation.doubleArray))
		lastAction = self.lastAction.intArray[0]

		newDiscState = self.getDiscState(observation.intArray)
		lastDiscState = self.getDiscState(self.lastObservation.intArray)
		newIntAction = self.getAction(newState, newDiscState)
		
		# Update eligibility traces
		phi_t = numpy.zeros(self.traces.shape)
		phi_tp = numpy.zeros(self.traces.shape)

		if self.basis is None:
			phi_tp[newDiscState, :,newIntAction] = newState
			phi_t[lastDiscState, :,lastAction] = lastState
		else:
			phi_tp[newDiscState, :,newIntAction] = self.basis.computeFeatures(newState)
			phi_t[lastDiscState, :,lastAction] = self.basis.computeFeatures(lastState)
		
		self.traces *= self.gamma * self.lmbda
		self.traces += phi_t

		self.update(phi_t, phi_tp, reward)

		returnAction=Action()
		returnAction.intArray=[newIntAction]
		
		self.lastAction=copy.deepcopy(returnAction)
		self.lastObservation=copy.deepcopy(observation)
		return returnAction


	def init_stepsize(self, weights_shape, params):
		self.step_sizes = numpy.ones(weights_shape) * self.alpha

	def compute_stepsize(self, phi_t, phi_tp, delta, reward):
		pass

	def update(self, phi_t, phi_tp, reward):
		# Compute Delta (TD-error)
		delta = numpy.dot(self.weights.flatten(), (self.gamma * phi_tp - phi_t).flatten()) + reward

		# Adaptive step-size if that is enabled
		self.compute_stepsize(phi_t, phi_tp, delta, reward)

		# Update the weights with both a scalar and vector stepsize used
		self.weights += self.step_sizes * delta * self.traces

	def agent_end(self,reward):
		"""Receive the final reward in an episode, also signaling the end of the episode.
		
		Args:
			reward: The reward received for taking the last action from the previous state.
		"""
		lastState = numpy.array(list(self.lastObservation.doubleArray))
		lastAction = self.lastAction.intArray[0]

		lastDiscState = self.getDiscState(self.lastObservation.intArray)

		# Update eligibility traces
		phi_t = numpy.zeros(self.traces.shape)
		phi_tp = numpy.zeros(self.traces.shape)

		if self.basis is None:
			phi_t[lastDiscState, :,lastAction] = lastState
		else:
			phi_t[lastDiscState, :,lastAction] = self.basis.computeFeatures(lastState)
		
		self.traces *= self.gamma * self.lmbda
		self.traces += phi_t
		self.update(phi_t, phi_tp, reward)

	def agent_cleanup(self):
		"""Perform any clean up operations before the end of an experiment."""
		pass
	
	def agent_message(self,inMessage):
		"""Receive a message from the environment or experiment and respond.
		
		Args:
			inMessage: A string message sent by either the environment or experiment to the agent.

		Returns:
			A string response message.
		"""
		return "SarsaLambda(Python) does not understand your message."


def addLinearTDArgs(parser):
	parser.add_argument("--epsilon", type=float, default=0.1, help="Probability of exploration with epsilon-greedy.")
	parser.add_argument("--softmax", type=float, help="Use softmax policies with the argument giving tau, the divisor which scales values used when computing soft-max policies.")
	parser.add_argument("--stepsize", "--alpha", type=float, default=0.01, help="The step-size parameter which affects how far in the direction of the gradient parameters are updated.")
	parser.add_argument("--adaptive_stepsize", choices=["ass", "autostep", "test"], help="Use an adaptive step-size algorithm.")
	parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor")
	parser.add_argument("--lambda", type=float, default=0.7, help="The eligibility traces decay rate. Set to 0 to disable eligibility traces.", dest='lmbda')
	parser.add_argument("--basis", choices=["trivial", "fourier", "tile", "rbf"], default="trivial", help="Set the basis to use for linear function approximation.")
	parser.add_argument("--autostep_mu", type=float, default=1.0e-2, help="Mu parameter for the Autostep algorithm. This is the meta-stepsize.")
	parser.add_argument("--autostep_tau", type=float, default=1.0e4, help="Tau parameter for the Autostep algorithm.")
	parser.add_argument("--fourier_order", type=int, default=3, help="Order for Fourier basis")
	parser.add_argument("--rbf_num", type=int, default=10, help="Number of radial basis functions to use.")
	parser.add_argument("--rbf_beta", type=float, default=1.0, help="Beta parameter for radial basis functions.")
	parser.add_argument("--tiles_num", type=int, default=100, help="Number of tilings to use with Tile Coding.")
	parser.add_argument("--tiles_size", type=int, default=2048, help="Memory size, number of weights, to use with Tile Coding.")

if __name__=="__main__":
	import argparse
	parser = argparse.ArgumentParser(description='Run SarsaLambda agent in network mode with linear function approximation.')
	addLinearTDArgs(parser)
	args = parser.parse_args()
	params = {}
	params['name'] = args.basis
	params['order'] = args.fourier_order
	params['num_functions'] = args.rbf_num
	params['beta'] = args.rbf_beta
	params['num_tiles'] = args.tiles_num
	params['num_weights'] = args.tiles_size
	alpha = args.stepsize

	epsilon = args.epsilon
	softmax = False
	if args.softmax is not None:
		softmax = True
		epsilon = args.softmax

	if args.adaptive_stepsize == "autostep":
		class AutoSarsa(stepsizes.Autostep, sarsa_lambda):
			def __init__(self, epsilon, alpha, gamma, lmbda, params={}, softmax=False):
				sarsa_lambda.__init__(self, epsilon, alpha, gamma, lmbda, params=params, softmax=softmax)

		params['autostep_mu'] = args.autostep_mu
		params['autostep_tau'] = args.autostep_tau
		AgentLoader.loadAgent(AutoSarsa(epsilon, alpha, args.gamma, args.lmbda, params=params, softmax=softmax))
	elif args.adaptive_stepsize == "ass":
		class ABSarsa(stepsizes.AlphaBounds, sarsa_lambda):
			def __init__(self, epsilon, alpha, gamma, lmbda, params={}, softmax=False):
				sarsa_lambda.__init__(self, epsilon, alpha, gamma, lmbda, params=params, softmax=softmax)
		AgentLoader.loadAgent(ABSarsa(epsilon, alpha, args.gamma, args.lmbda, params=params, softmax=softmax))
	else:
		AgentLoader.loadAgent(sarsa_lambda(epsilon, alpha, args.gamma, args.lmbda, params=params, softmax=softmax))
