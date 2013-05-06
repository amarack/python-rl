
from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.utils import TaskSpecVRLGLUE3
from pyrl.rlglue.registry import register_agent

from random import Random
import numpy
import copy

import pyrl.basis.fourier as fourier
import pyrl.basis.rbf as rbf
import pyrl.basis.tilecode as tilecode
import stepsizes

@register_agent
class sarsa_lambda(Agent):
	name = "Sarsa"

	def __init__(self, **kwargs):
		"""Initialize Sarsa based agent, or some subclass with given named parameters.
		
		Args:
			epsilon=0.1: Exploration rate for epsilon-greedy, or the rescale factor for soft-max policies.
			alpha=0.01: Step-Size for parameter updates.
			gamma=1.0: Discount factor for learning, also viewed as a planning/learning horizon.
			lmbda=0.7: Eligibility decay rate.
			softmax=False: True to use soft-max style policies, false to use epsilon-greedy policies.
			basis='trivial': Name of basis functions to use. [trivial, fourier, rbf, tile]
			fourier_order=3: Order of fourier basis to use if using fourier basis.
			rbf_number=0: Number of radial basis functions to use if doing rbf basis. Defaults to 0 for dim of states.
			rbf_beta=1.0: Beta parameter for rbf basis.
			tile_number=100: Number of tilings to use with tile coding basis.
			tile_weights=2048: Number of weights to use with tile coding.
			**kwargs: Additional named arguments

		"""

		self.randGenerator = Random()	
		self.lastAction=Action()
		self.lastObservation=Observation()
		self.params = kwargs

	def init_parameters(self):
		# Initialize algorithm parameters
		self.epsilon = self.params.setdefault('epsilon', 0.1)
		self.alpha = self.params.setdefault('alpha', 0.01)
		self.lmbda = self.params.setdefault('lmbda', 0.7)
		self.gamma = self.params.setdefault('gamma', 1.0)
		self.fa_name = self.params.setdefault('basis', 'trivial')
		self.softmax = self.params.setdefault('softmax', False)
		self.basis = None

	def randomize_parameters(self, **args):
		"""Generate parameters randomly, constrained by given named parameters.

		If used, this must be called before agent_init in order to have desired effect.
		
		Parameters that fundamentally change the algorithm are not randomized over. For 
		example, basis and softmax fundamentally change the domain and have very few values 
		to be considered. They are not randomized over.

		Basis parameters, on the other hand, have many possible values and ARE randomized. 

		Args:
			**args: Named parameters to fix, which will not be randomly generated

		Returns:
			List of resulting parameters of the class. Will always be in the same order. 
			Empty list if parameter free.

		"""

		# Randomize main parameters
		self.epsilon = args.setdefault('epsilon', numpy.random.random())
		self.alpha = args.setdefault('alpha', numpy.random.random())
		self.gamma = args.setdefault('gamma', numpy.random.random())
		self.lmbda = args.setdefault('lmbda', numpy.random.random())
		self.softmax = args.setdefault('softmax', self.softmax)
		self.fa_name = args.setdefault('basis', self.fa_name)
		param_list = [self.epsilon, self.alpha, self.gamma, self.lmbda, int(self.softmax)]

		# Randomize basis parameters
		if self.fa_name == 'fourier':
			self.params['fourier_order'] = args.setdefault('fourier_order', numpy.random.choice([3,5,7,9]))
			param_list.append(self.params['fourier_order'])
		elif self.fa_name == 'rbf':
			self.params['rbf_number'] = args.setdefault('rbf_number', numpy.random.randint(100))
			self.params['rbf_beta'] = args.setdefault('rbf_beta', numpy.random.random())
			param_list += [self.params['rbf_number'], self.params['rbf_beta']]
		elif self.fa_name == 'tile':
			self.params['tile_number'] = args.setdefault('tile_number', numpy.random.randint(200))
			self.params['tile_weights'] = args.setdefault('tile_weights', 2**numpy.random.randint(15))
			param_list += [self.params['tiles_number'], self.params['tiles_weights']]
		
		return param_list

	def agent_supported(self, parsedSpec):
		if parsedSpec.valid:
			# Check observation form, and then set up number of features/states
			assert len(parsedSpec.getDoubleObservations()) + len(parsedSpec.getIntObservations()) > 0, "Expecting at least one continuous or discrete observation"

			# Check action form, and then set number of actions
			assert len(parsedSpec.getIntActions())==1, "Expecting 1-dimensional discrete actions"
			assert len(parsedSpec.getDoubleActions())==0, "Expecting no continuous actions"
			assert not parsedSpec.isSpecial(parsedSpec.getIntActions()[0][0]), "Expecting min action to be a number not a special value"
			assert not parsedSpec.isSpecial(parsedSpec.getIntActions()[0][1]), "Expecting max action to be a number not a special value"
			return True
		else:
			return False

	def agent_init(self,taskSpec):
		"""Initialize the RL agent.

		Args:
			taskSpec: The RLGlue task specification string.
		"""

		# (Re)initialize parameters (incase they have been changed during a trial
		self.init_parameters()
		# Parse the task specification and set up the weights and such
		TaskSpec = TaskSpecVRLGLUE3.TaskSpecParser(taskSpec)
		if self.agent_supported(TaskSpec):
			self.numStates=len(TaskSpec.getDoubleObservations())
			self.discStates = numpy.array(TaskSpec.getIntObservations())
			self.numDiscStates = int(reduce(lambda a, b: a * (b[1] - b[0] + 1), self.discStates, 1.0)) 
			self.numActions=TaskSpec.getIntActions()[0][1]+1;

			if self.numStates == 0:
				# Only discrete states
				self.numStates = 1
				if self.fa_name != "trivial":
					print "Selected basis requires at least one continuous feature. Using trivial basis."
					self.fa_name = "trivial"

			# Set up the function approximation
			if self.fa_name == 'fourier':
				self.basis = fourier.FourierBasis(self.numStates, 
								  self.params.setdefault('fourier_order', 3), 
								  TaskSpec.getDoubleObservations())
				self.weights = numpy.zeros((self.numDiscStates, self.basis.numTerms, self.numActions))
			elif self.fa_name == 'rbf':
				num_functions = self.numStates if self.params.setdefault('rbf_number', 0) == 0 else self.params['rbf_number']
				self.basis = rbf.RBFBasis(self.numStates,
							  num_functions, 
							  self.params.setdefault('rbf_beta', 0.9), 
							  TaskSpec.getDoubleObservations())
				self.weights = numpy.zeros((self.numDiscStates, num_functions, self.numActions))
			elif self.fa_name == 'tile':
				self.basis = tilecode.TileCodingBasis(self.params.setdefault('tile_number', 100), 
								      self.params.setdefault('tile_weights', 2048))
				self.weights = numpy.zeros((self.numDiscStates, self.params['tile_weights'], self.numActions))
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

	def update_traces(self, phi_t, phi_tp):
		self.traces *= self.gamma * self.lmbda
		self.traces += phi_t
		
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
		phi_t[lastDiscState, :, lastAction] = lastState if self.basis is None else self.basis.computeFeatures(lastState)
		phi_tp[newDiscState, :, newIntAction] = newState if self.basis is None else self.basis.computeFeatures(newState)
		
		self.update_traces(phi_t, phi_tp)
		self.update(phi_t, phi_tp, reward)

		returnAction=Action()
		returnAction.intArray=[newIntAction]
		
		self.lastAction=copy.deepcopy(returnAction)
		self.lastObservation=copy.deepcopy(observation)
		return returnAction


	def init_stepsize(self, weights_shape, params):
		self.step_sizes = numpy.ones(weights_shape) * self.alpha

	def rescale_update(self, phi_t, phi_tp, delta, reward, descent_direction):
		return self.step_sizes * descent_direction

	def update(self, phi_t, phi_tp, reward):
		# Compute Delta (TD-error)
		delta = numpy.dot(self.weights.flatten(), (self.gamma * phi_tp - phi_t).flatten()) + reward

		# Update the weights with both a scalar and vector stepsize used
		# Adaptive step-size if that is enabled
		self.weights += self.rescale_update(phi_t, phi_tp, delta, reward, delta*self.traces)

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
		phi_t[lastDiscState, :, lastAction] = lastState if self.basis is None else self.basis.computeFeatures(lastState)

		self.update_traces(phi_t, phi_tp)
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
		return name + " does not understand your message."


@register_agent
class residual_gradient(sarsa_lambda):
	name = "Residual Gradient"
	def update_traces(self, phi_t, phi_tp):
		self.traces *= self.gamma * self.lmbda
		self.traces += (phi_t - self.gamma * phi_tp)


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
	params['alpha'] = args.stepsize
	params['gamma'] = args.gamma
	params['lmbda'] = args.lmbda

	if args.softmax is not None:
		params['softmax'] = True
		params['epsilon'] = args.softmax
	else:
		params['softmax'] = False
		params['epsilon'] = args.epsilon

	params['basis'] = args.basis
	params['fourier_order'] = args.fourier_order
	params['rbf_number'] = args.rbf_num
	params['rbf_beta'] = args.rbf_beta
	params['tile_number'] = args.tiles_num
	params['tile_weights'] = args.tiles_size

	if args.adaptive_stepsize == "autostep":
		AutoSarsa = stepsizes.genAdaptiveAgent(stepsizes.Autostep, sarsa_lambda)
		params['autostep_mu'] = args.autostep_mu
		params['autostep_tau'] = args.autostep_tau
		AgentLoader.loadAgent(AutoSarsa(**params))
	elif args.adaptive_stepsize == "ass":
		ABSarsa = stepsizes.genAdaptiveAgent(stepsizes.AlphaBounds, sarsa_lambda)
		AgentLoader.loadAgent(ABSarsa(**params))
	else:
		AgentLoader.loadAgent(sarsa_lambda(**params))
