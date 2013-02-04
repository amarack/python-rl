
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
import sarsa_lambda

class qlearning_agent(sarsa_lambda.sarsa_lambda):

	def __init__(self, epsilon, alpha, gamma, lmbda, params={}, softmax=False):
		sarsa_lambda.sarsa_lambda.__init__(self, epsilon, alpha, gamma, lmbda, params, softmax)

	
	def agent_step(self,reward, observation):
		newState = numpy.array(observation.doubleArray.tolist())
		lastState = numpy.array(self.lastObservation.doubleArray.tolist())
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
		if self.use_autostep:
			self.do_autostep(phi_t.flatten(), delta)
		if self.use_ass:
			phi_tp = numpy.zeros(phi_t.shape)
			if s_tp is not None:
				phi_tp[discState,:,a_tp] = s_tp
			self.do_adaptivestep(phi_t, phi_tp, delta)
		
		# Update the weights with both a scalar and vector stepsize used
		# (Maybe we should actually make them both work together naturally)
		self.weights += self.step_sizes * self.alpha * delta * self.traces
	
	def agent_end(self,reward):
		lastState = numpy.array(self.lastObservation.doubleArray.tolist())
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
	
	def agent_message(self,inMessage):
		return "Q-Lambda(Python) does not understand your message."

if __name__=="__main__":
	import argparse
	parser = argparse.ArgumentParser(description='Run Q-Lambda, q-learning with eligibility traces, agent in network mode with linear function approximation.')
	addLinearTDArgs(parser)
	args = parser.parse_args()
	params = {}
	params['autostep_mu'] = args.autostep_mu
	params['autostep_tau'] = args.autostep_tau
	params['name'] = args.basis
	params['order'] = args.fourier_order
	params['num_functions'] = args.rbf_num
	params['beta'] = args.rbf_beta
	params['num_tiles'] = args.tiles_num
	params['num_weights'] = args.tiles_size
	alpha = args.stepsize

	if args.adaptive_stepsize == "ass":
		alpha = "ass"
	elif args.adaptive_stepsize == "autostep":
		alpha = "autostep"

	epsilon = args.epsilon
	softmax = False
	if args.softmax is not None:
		softmax = True
		epsilon = args.softmax

	AgentLoader.loadAgent(qlearning_agent(epsilon, alpha, args.gamma, args.lmbda, params=params, softmax=softmax))
