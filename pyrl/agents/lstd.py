
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
import stepsizes

class LSTD(sarsa_lambda.sarsa_lambda):

	def __init__(self, epsilon, gamma, lmbda, params={}, softmax=False, online=False):
		sarsa_lambda.sarsa_lambda.__init__(self, epsilon, 1.0, gamma, lmbda, params, softmax)
		self.A = None
		self.B = None
		self.b = None
		self.online = online

	def init_stepsize(self, weights_shape, params):
		# LSTD(lambda)
		# Using step_sizes for b
		# Using traces for z
		self.A = numpy.zeros((numpy.prod(weights_shape),numpy.prod(weights_shape)))
		self.step_sizes = numpy.zeros((numpy.prod(weights_shape),))
		self.lstd_counter = 0

	def agent_start(self,observation):
		theState = numpy.array(list(observation.doubleArray))
		thisIntAction=self.getAction(theState, self.getDiscState(observation.intArray))
		returnAction=Action()
		returnAction.intArray=[thisIntAction]

		self.traces.fill(0.)

		self.lastAction=copy.deepcopy(returnAction)
		self.lastObservation=copy.deepcopy(observation)
		
		return returnAction
	
	def agent_step(self,reward, observation):
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
		
		self.update(phi_t, phi_tp, reward)
		
		# QLearning can choose action after update
		returnAction=Action()
		returnAction.intArray=[newIntAction]
		
		self.lastAction=copy.deepcopy(returnAction)
		self.lastObservation=copy.deepcopy(observation)
		return returnAction

	def shouldUpdate(self):
		self.lstd_counter += 1
		return self.online or self.lstd_counter % 100

	def update(self, phi_t, phi_tp, reward):
		#LSTD
		# z update..
		self.traces *= self.lmbda
		self.traces += phi_t

		# A update...
		d = phi_t.flatten() - phi_tp.flatten()
		self.A = self.A + numpy.outer(self.traces.flatten(), d)
		self.step_sizes += self.traces.flatten() * reward
		
		if self.shouldUpdate():
			if not self.online or self.lstd_counter == 1:
				self.B = numpy.linalg.pinv(self.A)
			else:
				self.B = matrix.SMInv(self.B, d, phi_t.flatten(), 1.)

			self.weights = numpy.dot(self.B, self.step_sizes).reshape(self.weights.shape)

	
	def agent_end(self,reward):
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
		
		self.update(phi_t, phi_tp, reward)
	
	def agent_message(self,inMessage):
		return "LSTD(lambda) [Python] does not understand your message."


class iLSTD(LSTD):
	def __init__(self, epsilon, alpha, gamma, lmbda, num_sweeps, params={}, softmax=False):
		sarsa_lambda.sarsa_lambda.__init__(self, epsilon, alpha, gamma, lmbda, params, softmax)
		self.A = None
		self.b = None
		self.num_sweeps = num_sweeps

	def update(self, phi_t, phi_tp, reward):
		#iLSTD
		# z update..
		self.traces *= self.lmbda
		self.traces += phi_t

		# A update...
		d = numpy.outer(self.traces.flatten(), phi_t - phi_tp)
		self.A = self.A + d
		self.step_sizes += self.traces.flatten() * reward - numpy.dot(d, self.weights.flatten())

		for i in range(self.num_sweeps):
			j = numpy.abs(self.step_sizes).argmax()
			self.weights.flat[j] += self.alpha * self.step_sizes[j]
			self.step_sizes -= self.alpha * self.step_sizes[j] * self.A[:,j]

	def shouldUpdate(self):
		return True



if __name__=="__main__":
	import argparse
	parser = argparse.ArgumentParser(description='Run Least Squares Temporal Difference Learning agent.')
	parser.add_argument("--epsilon", type=float, default=0.1, help="Probability of exploration with epsilon-greedy.")
	parser.add_argument("--softmax", type=float, help="Use softmax policies with the argument giving tau, the divisor which scales values used when computing soft-max policies.")
	parser.add_argument("--stepsize", "--alpha", type=float, default=0.01, help="The step-size parameter which affects how far in the direction of the gradient parameters are updated. Only with iLSTD.")
	parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor")
	parser.add_argument("--lambda", type=float, default=0.7, help="The eligibility traces decay rate. Set to 0 to disable eligibility traces.", dest='lmbda')
	parser.add_argument("--num_sweeps", type=int, default=1, help="Number of sweeps to perform per step in iLSTD.")
	parser.add_argument("--algorithm", choices=["lstd", "online", "ilstd", "rlstd"], default="lstd", help="Set the LSTD algorithm to use. LSTD, Online LSTD, iLSTD, or Recursive LSTD.")
	parser.add_argument("--basis", choices=["trivial", "fourier", "tile", "rbf"], default="trivial", help="Set the basis to use for linear function approximation.")
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
		AgentLoader.loadAgent(LSTD(epsilon, args.gamma, args.lmbda, params=params, softmax=softmax, online=False))
	elif args.algorithm == "online":
		AgentLoader.loadAgent(LSTD(epsilon, args.gamma, args.lmbda, params=params, softmax=softmax, online=True))
	elif args.algorithm == "ilstd":
		AgentLoader.loadAgent(iLSTD(epsilon, alpha, args.gamma, args.lmbda, args.num_sweeps, params=params, softmax=softmax))
	elif args.algorithm == "rlstd":
		print "Not yet implemented. Coming soon!"
	else:
		print "Error: Unknown LSTD algorithm", args.algorithm
