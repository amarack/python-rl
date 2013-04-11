# Author: Will Dabney

from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.utils import TaskSpecVRLGLUE3

from random import Random
import numpy
import sys
import copy

import pyrl.misc.matrix as matrix
import pyrl.basis.fourier as fourier
import pyrl.basis.rbf as rbf
import pyrl.basis.tilecode as tilecode

import sarsa_lambda

class LSTD(sarsa_lambda.sarsa_lambda):
	"""Least Squares Temporal Difference Learning (LSTD) agent."""
	
	def __init__(self, epsilon, gamma, lmbda, params={}, softmax=False):
		"""Initializes LSTD with exploration rate, discount factor, and eligibility trace decay rate.
		
		Args:
			epsilon: The exploration rate for epsilon-greedy.
			gamma: The discount factor for the domain
			lmbda: The (lambda) eligibility trace decay rate
			params: Additional parameters
			softmax: Whether to switch from eps-greedy to softmax policies. Default to False.
		"""
		# lstd_gamma holds the true gamma, and we pass 1. for gamma to sarsa, so we 
		# don't need to rewrite the eligibility traces update.
		sarsa_lambda.sarsa_lambda.__init__(self, epsilon, 1.0, 1.0, lmbda, params, softmax)
		self.A = None
		self.b = None
		self.lstd_gamma = gamma

	def init_stepsize(self, weights_shape, params):
		"""Initializes the step-size variables, in this case meaning the A matrix and b vector.

		Args:
			weights_shape: Shape of the weights array
			params: Additional parameters.
		"""
		# Using step_sizes for b
		# Using traces for z
		self.A = numpy.zeros((numpy.prod(weights_shape),numpy.prod(weights_shape)))
		self.step_sizes = numpy.zeros((numpy.prod(weights_shape),))
		self.lstd_counter = 0
	
	def shouldUpdate(self):
		self.lstd_counter += 1
		return self.lstd_counter % 100

	def update(self, phi_t, phi_tp, reward):
		# z update..
		self.traces *= self.lmbda
		self.traces += phi_t

		# A update...
		d = phi_t.flatten() - self.lstd_gamma * phi_tp.flatten()
		self.A = self.A + numpy.outer(self.traces.flatten(), d)
		self.step_sizes += self.traces.flatten() * reward
		
		if self.shouldUpdate():
			B = numpy.linalg.pinv(self.A)
			self.weights = numpy.dot(B, self.step_sizes).reshape(self.weights.shape)
		
	def agent_message(self,inMessage):
		"""Receive a message from the environment or experiment and respond.
		
		Args:
			inMessage: A string message sent by either the environment or experiment to the agent.

		Returns:
			A string response message.
		"""
		return "LSTD(lambda) [Python] does not understand your message."


class oLSTD(LSTD):
	"""Online Least Squares Temporal Difference Learning (oLSTD) agent. O(n^2) time complexity."""
	def __init__(self, epsilon, alpha, gamma, lmbda, params={}, softmax=False):
		"""Initializes LSTD with exploration rate, discount factor, and eligibility trace decay rate.
		
		Args:
			epsilon: The exploration rate for epsilon-greedy.
			alpha: The multiplicative scale for the random initialization of the inverse matrix.
			gamma: The discount factor for the domain
			lmbda: The (lambda) eligibility trace decay rate
			params: Additional parameters
			softmax: Whether to switch from eps-greedy to softmax policies. Default to False.
		"""
		sarsa_lambda.sarsa_lambda.__init__(self, epsilon, alpha, 1.0, lmbda, params, softmax)
		self.A = None
		self.b = None
		self.lstd_gamma = gamma

	def init_stepsize(self, weights_shape, params):
		"""Initializes the step-size variables, in this case meaning the A matrix and b vector.

		Args:
			weights_shape: Shape of the weights array
			params: Additional parameters.
		"""
		self.A = numpy.eye(numpy.prod(weights_shape)) 
		self.A += numpy.random.random(self.A.shape)*self.alpha
		self.step_sizes = numpy.zeros((numpy.prod(weights_shape),))
		self.lstd_counter = 0

	def update(self, phi_t, phi_tp, reward):
		self.traces *= self.lmbda
		self.traces += phi_t

		d = phi_t.flatten() - self.lstd_gamma * phi_tp.flatten()
		self.step_sizes += self.traces.flatten() * reward

		self.A = matrix.SMInv(self.A, self.traces.flatten(), d, 1.)
		self.weights = numpy.dot(self.A, self.step_sizes).reshape(self.weights.shape)

	def shouldUpdate(self):
		return True

class iLSTD(LSTD):
	"""Incremental Least Squares Temporal Difference Learning (iLSTD) agent."""
	def __init__(self, epsilon, alpha, gamma, lmbda, num_sweeps, params={}, softmax=False):
		"""Initializes iLSTD with exploration rate, step-size, discount factor, and eligibility trace decay rate.
		
		Args:
			epsilon: The exploration rate for epsilon-greedy.
			alpha: The step-size to use.
			gamma: The discount factor for the domain
			lmbda: The (lambda) eligibility trace decay rate
			num_sweeps: The number of sweeps to perform per time step.
			params: Additional parameters
			softmax: Whether to switch from eps-greedy to softmax policies. Default to False.
		"""
		sarsa_lambda.sarsa_lambda.__init__(self, epsilon, alpha, 1.0, lmbda, params, softmax)
		self.A = None
		self.b = None
		self.num_sweeps = num_sweeps
		self.lstd_gamma = gamma

	def update(self, phi_t, phi_tp, reward):
		#iLSTD
		# z update..
		self.traces *= self.lmbda
		self.traces += phi_t

		# A update...
		d = numpy.outer(self.traces.flatten(), phi_t.flatten() - self.lstd_gamma*phi_tp.flatten())
		self.A = self.A + d
		self.step_sizes += self.traces.flatten() * reward - numpy.dot(d, self.weights.flatten())
		for i in range(self.num_sweeps):
			j = numpy.abs(self.step_sizes).argmax()
			self.weights.flat[j] += self.alpha * self.step_sizes[j]
			self.step_sizes -= self.alpha * self.step_sizes[j] * self.A.T[:,j]

	def shouldUpdate(self):
		return True


class RLSTD(LSTD):
	# alpha is the forgetting factor, delta is what to initialize A to
	def __init__(self, epsilon, alpha, delta, gamma, lmbda, params={}, softmax=False):
		sarsa_lambda.sarsa_lambda.__init__(self, epsilon, alpha, gamma, lmbda, params, softmax)
		self.delta = delta
		self.A = None

	def init_stepsize(self, weights_shape, params):
		self.A = numpy.eye(numpy.prod(weights_shape)) * self.delta 

	def update(self, phi_t, phi_tp, reward):
		#RLS-TD(lambda)
		self.traces *= self.lmbda * self.gamma
		self.traces += phi_t

		# A update...
		d = numpy.dot(self.A, self.traces.flatten()) 
		K = d / (self.alpha + numpy.dot((phi_t - self.gamma * phi_tp).flatten(), d))
		self.A = matrix.SMInv(self.A, self.traces.flatten(), (phi_t - self.gamma*phi_tp).flatten(), self.alpha)
		self.weights += (reward - numpy.dot((phi_t - self.gamma * phi_tp).flatten(), self.weights.flatten())) * K.reshape(self.weights.shape)

	def shouldUpdate(self):
		return True



if __name__=="__main__":
	import argparse
	parser = argparse.ArgumentParser(description='Run Least Squares Temporal Difference Learning agent.')
	parser.add_argument("--epsilon", type=float, default=0.1, help="Probability of exploration with epsilon-greedy.")
	parser.add_argument("--softmax", type=float, 
			    help="Use softmax policies with the argument giving tau, the divisor which scales values used when computing soft-max policies.")
	parser.add_argument("--stepsize", "--alpha", type=float, default=0.01, 
			    help="The step-size parameter which affects how far in the direction of the gradient parameters are updated. Only with iLSTD.")
	parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor")
	parser.add_argument("--mu", type=float, default=1., help="Forgetting factor for RLS-TD")
	parser.add_argument("--beta", type=float, default=0.01, help="Online LSTD initializes A^-1 to I + beta*RandMatrix.")
	parser.add_argument("--lambda", type=float, default=0.7, help="The eligibility traces decay rate. Set to 0 to disable eligibility traces.", dest='lmbda')
	parser.add_argument("--num_sweeps", type=int, default=1, help="Number of sweeps to perform per step in iLSTD.")
	parser.add_argument("--delta", type=float, default=200., 
			    help="Value to initialize diagonal matrix to, for inverse matrix, in RLS-TD.")
	parser.add_argument("--algorithm", choices=["lstd", "online", "ilstd", "rlstd"], 
			    default="lstd", help="Set the LSTD algorithm to use. LSTD, Online LSTD, iLSTD, or Recursive LSTD.")
	parser.add_argument("--basis", choices=["trivial", "fourier", "tile", "rbf"], default="trivial", 
			    help="Set the basis to use for linear function approximation.")
	parser.add_argument("--fourier_order", type=int, default=3, help="Order for Fourier basis")
	parser.add_argument("--rbf_num", type=int, default=10, help="Number of radial basis functions to use.")
	parser.add_argument("--rbf_beta", type=float, default=1.0, help="Beta parameter for radial basis functions.")
	parser.add_argument("--tiles_num", type=int, default=100, help="Number of tilings to use with Tile Coding.")
	parser.add_argument("--tiles_size", type=int, default=2048, help="Memory size, number of weights, to use with Tile Coding.")

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

	if args.algorithm == "lstd":
		AgentLoader.loadAgent(LSTD(epsilon, args.gamma, args.lmbda, params=params, softmax=softmax))
	elif args.algorithm == "online":
		AgentLoader.loadAgent(oLSTD(epsilon, args.beta, args.gamma, args.lmbda, params=params, softmax=softmax))
	elif args.algorithm == "ilstd":
		AgentLoader.loadAgent(iLSTD(epsilon, alpha, args.gamma, args.lmbda, args.num_sweeps, params=params, softmax=softmax))
	elif args.algorithm == "rlstd":
		AgentLoader.loadAgent(RLSTD(epsilon, args.mu, args.delta, args.gamma, args.lmbda, params=params, softmax=softmax))
	else:
		print "Error: Unknown LSTD algorithm", args.algorithm
