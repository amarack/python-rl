
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
		self.gamma = self.params.setdefault('gamma', 0.99)
		self.params.setdefault('num_iterations', 30)
		fa_name = self.params.setdefault('basis', 'trivial')
		self.ranges, self.actions = model.getStateSpace()
		self.has_plan = False

		# Set up basis
		if fa_name == 'fourier':
			# number of dimensions, order, and then the dimension ranges
			self.basis = fourier.FourierBasis(len(self.ranges),
							  self.params.setdefault('fourier_order', 3), 
							  self.ranges)
		elif fa_name == 'rbf':
			self.basis = rbf.RBFBasis(len(self.ranges),
						  self.params.setdefault('rbf_num_functions', len(self.ranges)), 
						  self.params.setdefault('rbf_beta', 1.0), 
						  self.ranges)
		elif fa_name == 'tile':
			self.basis = tilecode.TileCodingBasis(self.params.setdefault('tile_numtiles', 100), 
							      self.params.setdefault('tile_numweights', 2048))
		else:
			self.basis = None

		# Set up regressor
		learn_name = self.params.setdefault('regressor', 'lstsqr')
		if learn_name == 'lstsqr':
			#self.learner = linear_model.LinearRegression()
			self.learner = linear_model.Ridge (alpha = .2)
		else:
			self.learner = None
		
	def getStateAction(self, state, action):
		if self.basis is not None:
			state = self.basis.computeFeatures(state)

		stateaction = numpy.zeros((self.actions, len(state)))
		stateaction[action,:] = state
		return stateaction.flatten()

	def predict(self, state, action):
		if self.model.has_fit[action]:
			return self.model.predict(state, action)
		else:
			return None
	
	def getValue(self, state):
		if self.has_plan:
			return self.learner.predict([self.getStateAction(state, a) for a in range(self.actions)])

	def getAction(self, state):
		if self.has_plan:
			return self.learner.predict([self.getStateAction(state, a) for a in range(self.actions)]).argmax()
		else:
			return self.randGenerator.randint(0, self.actions-1)
		
	
        def updatePlan(self):
		samples = self.model.sampleStateActions(self.params.setdefault('support_size', 1000))
		# Fitted Q-Iteration
		# predict r + gamma * max Q(s', a') for each s,a
		# for each action, tuple containing for each sample: s', reward, terminates
		outcomes = self.model.predictSet(samples)
		Xp = []
		X = []
		R = []
		gammas = []
		A = []
		S = []
			
		for a in range(self.actions):
			Xp += map(lambda k: [self.getStateAction(k, b) for b in range(self.actions)], outcomes[a][0])
			X += map(lambda k: self.getStateAction(k, a), samples[a])
			R += list(outcomes[a][1])
			A += [a]*len(outcomes[a][1])
			S += list(samples[a])
			gammas += list((1.0 - outcomes[a][2]) * self.gamma)

		Xp = numpy.array(Xp)
		Xp = Xp.reshape(Xp.shape[0]*Xp.shape[1], Xp.shape[2])
		X = numpy.array(X)
		R = numpy.array(R)
		gammas = numpy.array(gammas)
		targets = []
		self.has_plan = False
		for iter in range(self.params['num_iterations']):
			if self.has_plan:
				Qprimes = self.learner.predict(Xp).reshape((X.shape[0], self.actions))
				targets = R + gammas*Qprimes.max(1)
			else:
				targets = R
				self.has_plan = True

			self.learner.fit(X, targets)


