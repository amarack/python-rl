
from random import Random
import numpy
import sys
import copy

import pyrl.basis.fourier as fourier
import pyrl.basis.rbf as rbf
import pyrl.basis.tilecode as tilecode
from planner import Planner

from sklearn import linear_model
from sklearn.svm import SVR
from sklearn import tree

# Fun Idea: For any situation where we run batch methods on a fundamentally incremental 
# problem, we should look at an exponential decay of the rate at which samples are included
# into our experiences used for training. 

# Fitted Q-Iteration is just like the Batch Modeler, 
# we can apply any batch regression algorithm:
#+    Linear least squares with some basis function
#+     Regression Trees
#+     SVMs
#     Gaussian Processes for Regression

class FittedQIteration(Planner):
	def __init__(self, gamma, model, params={}):
		Planner.__init__(self, gamma, model, params)

		self.randGenerator = Random()	
		self.ranges, self.actions = model.getStateSpace()
		self.has_plan = False
		self.params.setdefault('iterations', 200)
		self.params.setdefault('support_size', 200)

		# Set up basis
		self.basis = None
		fa_name = self.params.setdefault('basis', 'trivial')
		if fa_name == 'fourier':
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
		learn_name = self.params.setdefault('regressor', 'linreg')
		if learn_name == 'linreg':
			self.learner = linear_model.LinearRegression()
		elif learn_name == 'ridge': 
			self.learner = linear_model.Ridge(alpha = self.params.setdefault('l2', 0.5))
		elif learn_name == 'tree':
			self.learner = tree.DecisionTreeRegressor()
		elif learn_name == 'svm':
			self.learner = SVR()
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
			return None, None, None
	
	def getValue(self, state):
		if self.has_plan:
			return self.learner.predict([self.getStateAction(state, a) for a in range(self.actions)]).max()
		else:
			return None

	def getAction(self, state):
		if self.has_plan:
			return self.learner.predict([self.getStateAction(state, a) for a in range(self.actions)]).argmax()
		else:
			return self.randGenerator.randint(0, self.actions-1)
		
	# This is really ugly, probably also using much more memory than needed
	# TODO: clean it up
        def updatePlan(self, retry_on_fail=True):
		# Fitted Q-Iteration
		# predict r + gamma * max Q(s', a') for each s,a
		# for each action, tuple containing for each sample: s', reward, terminates		
		for sample_iter in range(self.params.setdefault('resample', 1)):
			self.has_plan = False
			prev_coef = None
			samples = self.model.sampleStateActions(self.params['support_size'])
			outcomes = self.model.predictSet(samples)
			kn = self.model.getKnown(samples)
			Xp = []
			X = []
			R = []
			gammas = []
			S = []
			K = []
			A = []
			for a in range(self.actions):
				for out in zip(samples[a], outcomes[a][0]):
					print "###", out[0], out[1], a
				S += list(samples[a])
				Xp += map(lambda k: [self.getStateAction(k, b) for b in range(self.actions)], outcomes[a][0])
				X += map(lambda k: self.getStateAction(k, a), samples[a])
				R += list(outcomes[a][1])
				gammas += list((1.0 - outcomes[a][2]) * self.gamma)
				K += list(kn[a])
				A += [a for k in samples[a]]
			Xp = numpy.array(Xp)
			Xp = Xp.reshape(Xp.shape[0]*Xp.shape[1], Xp.shape[2])
			X = numpy.array(X)
			R = numpy.array(R)
			gammas = numpy.array(gammas)
			targets = []
			S = numpy.array(S)
			Qp = None
			error = 1.0 
			iter2 = 0 
			threshold = 1.0e-4
			while error > threshold and iter2 < self.params['iterations']:
				if self.has_plan:
					Qprimes = self.learner.predict(Xp).reshape((X.shape[0], self.actions))
					targets = R + gammas*Qprimes.max(1)
					Qp = Qprimes
				else:
					targets = R
					self.has_plan = True

				self.learner.fit(X, targets)

				try:
					if prev_coef is not None:
						error = numpy.linalg.norm(prev_coef - self.learner.coef_)
					prev_coef = self.learner.coef_.copy()
				except:
					pass

				iter2 += 1

			print "#?", sample_iter, iter2, error, self.model.exp_index

			import csv
			with open("dv.dat", "wb") as f:
				csvw = csv.writer(f)
				csvw.writerows(map(lambda k: list(k[0]) + [k[1], k[2], k[3], k[5], k[4]], zip(S, R, K, gammas, targets, A)))
			if error <= threshold:
				return
