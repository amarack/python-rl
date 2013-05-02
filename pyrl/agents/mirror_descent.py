
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

import sarsa_lambda, qlearning
import stepsizes


def pnorm_linkfunc(weights, order):
	"""Link function induced by the p-norm."""
	return numpy.sign(weights) * numpy.abs(weights)**(order-1) / numpy.linalg.norm(weights, ord=order)**(order - 2)
		

@register_agent
class md_qlearn(qlearning.qlearning_agent):
	name = "Sparse Mirror Descent Q-Learning"
	
	def __init__(self, **kwargs):
		qlearning.qlearning_agent.__init__(self, **kwargs)
		self.sparsity = kwargs.setdefault('sparsity', 0.01)

	def agent_start(self,observation):
		returnAction = qlearning.qlearning_agent.agent_start(self, observation)
		self.pnorm = 2. * max(1, numpy.log10(numpy.prod(self.weights.shape)))
		self.qnorm = self.pnorm / (self.pnorm - 1.)
		return returnAction

	def proj_dual(self, weights):
		return pnorm_linkfunc(weights.flatten(), self.qnorm).reshape(weights.shape)

	def proj_primal(self, weights):
		return pnorm_linkfunc(weights.flatten(), self.pnorm).reshape(weights.shape)

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

		# Update dual weights
		dual_weights = self.proj_dual(self.weights)
		dual_weights += self.step_sizes * delta * self.traces
		
		# Truncate weights for sparsity
		dual_weights = numpy.sign(dual_weights) * (numpy.abs(dual_weights) - self.step_sizes * self.sparsity).clip(0.0)
		
		# Update the weights
		self.weights = self.proj_primal(dual_weights)

		
@register_agent
class md_sarsa(sarsa_lambda.sarsa_lambda):
	name = "Sparse Mirror Descent Sarsa"
	
	def __init__(self, **kwargs):
		sarsa_lambda.sarsa_lambda.__init__(self, **kwargs)
		self.sparsity = kwargs.setdefault('sparsity', 0.01)

	def agent_start(self,observation):
		returnAction = sarsa_lambda.sarsa_lambda.agent_start(self, observation)
		self.pnorm = 2. * max(1, numpy.log10(numpy.prod(self.weights.shape)))
		self.qnorm = self.pnorm / (self.pnorm - 1.)
		return returnAction

	def proj_dual(self, weights):
		return pnorm_linkfunc(weights.flatten(), self.qnorm).reshape(weights.shape)

	def proj_primal(self, weights):
		return pnorm_linkfunc(weights.flatten(), self.pnorm).reshape(weights.shape)

	def update(self, phi_t, phi_tp, reward):
		# Compute Delta (TD-error)
		delta = numpy.dot(self.weights.flatten(), (self.gamma * phi_tp - phi_t).flatten()) + reward

		# Update dual weights
		dual_weights = self.proj_dual(self.weights)
		dual_weights += self.step_sizes * delta * self.traces
		
		# Truncate weights for sparsity
		dual_weights = numpy.sign(dual_weights) * (numpy.abs(dual_weights) - self.step_sizes * self.sparsity).clip(0.0)
		
		# Update the weights
		self.weights = self.proj_primal(dual_weights)


# NOTE: This agent is not working at all. Not sure yet what is wrong
@register_agent
class cmd_sarsa(sarsa_lambda.sarsa_lambda):
	name = "Composite Mirror Descent Sarsa"
	
	def __init__(self, **kwargs):
		sarsa_lambda.sarsa_lambda.__init__(self, **kwargs)
		self.sparsity = kwargs.setdefault('sparsity', 0.01)
		self.covariance = None

	def update(self, phi_t, phi_tp, reward):
		# Compute Delta (TD-error)
		delta = numpy.dot(self.weights.flatten(), (self.gamma * phi_tp - phi_t).flatten()) + reward

		if self.covariance is None:
			self.covariance = numpy.zeros(phi_t.shape)

		self.covariance += phi_t**2

		H = numpy.sqrt(self.covariance)
		H[H == 0.0] = 1.0
		nobis_term = self.step_sizes / H

		# Update the weights
		update = self.weights - nobis_term * delta * self.traces
		self.weights = numpy.sign(update) * (numpy.abs(update) - nobis_term * self.sparsity)




