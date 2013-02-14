
from random import Random
import numpy
import sys
import copy

import pyrl.basis.fourier as fourier
import pyrl.basis.rbf as rbf
import pyrl.basis.tilecode as tilecode
from planner import Planner

from sklearn import linear_model

# Fun Idea: For any situation where we run batch methods on a fundamentally incremental 
# problem, we should look at an exponential decay of the rate at which samples are included
# into our experiences used for training. 

# Fitted Q-Iteration is just like the Batch Modeler, 
# we can apply any batch regression algorithm:
#     Linear least squares with some basis function
#     Regression Trees
#     Random Forrests and other ensemble methods with trees
#     SVMs
#     kNN with KD-Trees etc.
#     Gaussian Processes for Regression

class FittedQIteration(Planner):
	def __init__(self, model, params={}):
		Planner.__init__(self, model, params)
		self.randGenerator = Random()	
		self.basis = None
		self.gamma = self.params.setdefault('gamma', 0.999)
		fa_name = self.params.setdefault('basis', 'trivial')
		self.ranges, self.actions = model.getStateSpace()

		# Set up basis
		if fa_name == 'fourier':
			# number of dimensions, order, and then the dimension ranges
			self.basis = fourier.FourierBasis(len(ranges),
							  self.params.setdefault('fourier_order', 3), 
							  ranges)
		elif fa_name == 'rbf':
			self.basis = rbf.RBFBasis(len(ranges),
						  self.params.setdefault('rbf_num_functions', len(ranges)), 
						  self.params.setdefault('rbf_beta', 1.0), 
						  ranges)
		elif fa_name == 'tile':
			self.basis = tilecode.TileCodingBasis(self.params.setdefault('tile_numtiles', 100), 
							      self.params.setdefault('tile_numweights', 2048))
		else:
			self.basis = None

		# Set up regressor
		learn_name = self.params.setdefault('regressor', 'lstsqr')
		if learn_name == 'lstsqr':
			self.learner = linear_model.LinearRegression()
		else:
			self.learner = None
		
	def getStateAction(self, state, action):
		if self.basis is not None:
			state = self.basis.computeFeatures(state)
		
		stateaction = numpy.zeros((self.actions, len(state)))
		stateaction[action,:] = state
		return stateaction.flatten()

	def getAction(self, state):
		return self.learner.predict([self.getStateAction(state, a) for a in range(self.actions)]).argmax()
		
        def updatePlan(self):
		samples = self.model.sampleStateAction(self.params.setdefault('support_size', 1000))
		# Fitted Q-Iteration
		# predict r + gamma * max Q(s', a') for each s,a
		# for each action, tuple containing for each sample: s', reward, terminates
		outcomes = self.model.predictSet(samples)
		Xp = []
		X = []
		R = []
		gammas = []
		for a in range(self.actions):
			Xp += map(lambda k: self.getStateAction(k, a), outcomes[a][0])
			X += map(lambda k: self.getStateAction(k, a), samples[a])
			R += list(outcomes[a][1])
			gammas += list((outcomes[a][2] == 0) * self.gamma)

		Xp = numpy.array(Xp)
		X = numpy.array(X)
		R = numpy.array(R)
		gammas = numpy.array(gammas)
		targets = []

		for iter in range(10):
			Qprimes = self.learner.predict(Xp).max(1)
			targets = R + gammas*Qprimes
			self.learner.fit(X, targets)

		

if __name__=="__main__":
	import argparse
	parser = argparse.ArgumentParser(description='Run SarsaLambda agent in network mode with linear function approximation.')
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

	AgentLoader.loadAgent(sarsa_lambda(epsilon, alpha, args.gamma, args.lmbda, params=params, softmax=softmax))
