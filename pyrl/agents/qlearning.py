
# Author: Will Dabney

from random import Random
import numpy
import copy

from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.utils import TaskSpecVRLGLUE3
from pyrl.rlglue.registry import register_agent

import sarsa_lambda
import stepsizes

@register_agent
class qlearning_agent(sarsa_lambda.sarsa_lambda):
	name = "Q-Learning"
	
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

		# Update eligibility traces
		phi_t = numpy.zeros(self.traces.shape)

		if self.basis is None:
			phi_t[lastDiscState, :,lastAction] = lastState
		else:
			phi_t[lastDiscState, :,lastAction] = self.basis.computeFeatures(lastState)
		
		self.traces *= self.gamma * self.lmbda
		self.traces += phi_t

		self.update(phi_t, newState, newDiscState, reward)
		
		# QLearning can choose action after update
		newIntAction = self.getAction(newState, newDiscState)
		returnAction=Action()
		returnAction.intArray=[newIntAction]
		
		self.lastAction=copy.deepcopy(returnAction)
		self.lastObservation=copy.deepcopy(observation)
		return returnAction

	def update(self, phi_t, state, discState, reward):
		Q_tp = 0.0
		a_tp = 0
		s_tp = None
		if state is not None:
			# Find max_a Q(s_tp)
			if self.basis is None:
				s_tp = state
			else:
				s_tp = self.basis.computeFeatures(state)
			Q_tp = numpy.dot(self.weights[discState,:,:].T, s_tp)
			a_tp = Q_tp.argmax()
			Q_tp = Q_tp.max()

		# Compute Delta (TD-error)
		delta = self.gamma*Q_tp + reward - numpy.dot(self.weights.flatten(), phi_t.flatten())

		# Adaptive step-size if that is enabled
		phi_tp = numpy.zeros(phi_t.shape)
		if s_tp is not None:
			phi_tp[discState,:,a_tp] = s_tp

		# Update the weights with both a scalar and vector stepsize used
		# (Maybe we should actually make them both work together naturally)
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

		if self.basis is None:
			phi_t[lastDiscState, :,lastAction] = lastState
		else:
			phi_t[lastDiscState, :,lastAction] = self.basis.computeFeatures(lastState)
		
		self.traces *= self.gamma * self.lmbda
		self.traces += phi_t
		self.update(phi_t, None, 0, reward)
	
if __name__=="__main__":
	import argparse
	parser = argparse.ArgumentParser(description='Run Q-Lambda, q-learning with eligibility traces, agent in network mode with linear function approximation.')
	addLinearTDArgs(parser)
	args = parser.parse_args()
	params = {}
	params['alpha'] = args.alpha
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
		AutoQ = stepsizes.genAdaptiveAgent(stepsizes.Autostep, qlearning_agent)
		params['autostep_mu'] = args.autostep_mu
		params['autostep_tau'] = args.autostep_tau
		AgentLoader.loadAgent(AutoQ(**params))
	elif args.adaptive_stepsize == "ass":
		ABQ = stepsizes.genAdaptiveAgent(stepsizes.AlphaBounds, qlearning_agent)
		AgentLoader.loadAgent(ABQ(**params))
	else:
		AgentLoader.loadAgent(qlearning_agent(**params))

