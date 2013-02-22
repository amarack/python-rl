
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
		self.params.setdefault('num_iterations', 200) #100 was good
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
			self.learner = linear_model.Ridge (alpha = 0.5)
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
		

        def updatePlan(self):
		# Fitted Q-Iteration
		# predict r + gamma * max Q(s', a') for each s,a
		# for each action, tuple containing for each sample: s', reward, terminates
		self.has_plan = False
		prev_coef = None
		print "#,,,", self.model.exp_index
		sys.stdout.flush()
		for iter in range(1):#self.params['num_iterations']):
			samples = self.model.sampleStateActions(self.params.setdefault('support_size', 200))
			outcomes = self.model.predictSet(samples)
			Xp = []
			X = []
			R = []
			gammas = []
			A = []
			S = []
			Sp = []
			KN = []
			DS = []
			KNOWN=[]
			for a in range(self.actions):
				print "#numforAction", a, len(outcomes[a][0]), len(outcomes[a][1])
				Xp += map(lambda k: [self.getStateAction(k, b) for b in range(self.actions)], outcomes[a][0])
				X += map(lambda k: self.getStateAction(k, a), samples[a])
				R += list(outcomes[a][1])
				A += [a]*len(outcomes[a][1])
				S += list(samples[a])
				gammas += list((1.0 - outcomes[a][2]) * self.gamma)
				# DEBUGGING
				di, ind = self.model.model[a][0].kneighbors(self.model.normState(samples[a]))
				KN += map(lambda k: list(self.model.denormState(self.model.experiences[k[0],a])) + [self.model.terminates[k[0],a], k[0]], ind)
				DS += list(di)
				KNOWN += map(lambda k: self.model.isKnown(k,a), samples[a])

			Xp = numpy.array(Xp)
			Xp = Xp.reshape(Xp.shape[0]*Xp.shape[1], Xp.shape[2])
			X = numpy.array(X)
			R = numpy.array(R)
			S = numpy.array(S)
			gammas = numpy.array(gammas)
			targets = []
			
#		sys.exit(1)
			error = 1.0
			iter2 = 0
			while error > 1.0e-2 and iter2 < self.params['num_iterations']:
#			for iter2 in range(self.params['num_iterations']):
				if self.has_plan:
					Qprimes = self.learner.predict(Xp).reshape((X.shape[0], self.actions))
					Qprimes[numpy.where(Qprimes > 0)] = 0 ##??
					targets = R + gammas*Qprimes.max(1)
				else:
					targets = R
					self.has_plan = True

				self.learner.fit(X, targets)
				
				if prev_coef is not None:
					error = numpy.linalg.norm(prev_coef - self.learner.coef_)
					print "# COEF", error, self.getValue(numpy.array([0, -0.123, 0.00322]))
					sys.stdout.flush()
				prev_coef = self.learner.coef_.copy()
				iter2 += 1
				import csv
				with open("dd.dat", "wb") as f:
					csvr = csv.writer(f)
					csvr.writerows(map(lambda k: list(k[0]) + [k[1],k[2],k[3],k[4],k[6],list(k[5]),k[7]], zip(S, KNOWN, targets, R, gammas, KN,DS,A)))

#			csvr.writerow(R)
#			csvr.writerows(Xp)

#			print self.learner.coef_
#		dis, ind = self.model.model[0][0].kneighbors([numpy.array([0, 0.5, 0.0])])
#		print "Distances: ", dis
#		print "Weights: ", self.model.gaussianDist(dis)
		print "#", self.model.predict(numpy.array([0, 0.5, 0.0]), 1), self.model.model[1][0].predict([numpy.array([0,0.5, 0.0])])
		print "#", self.model.model[1][0].kneighbors([numpy.array([0,0.5, 0.0])]), self.model.experiences[self.model.model[1][0].kneighbors([numpy.array([0,0.5, 0.0])])[1][0][0], 1, :], \
		    self.model.terminates[self.model.model[1][0].kneighbors([numpy.array([0,0.5, 0.0])])[1][0][0], 1]
		print "#,,,", self.model.exp_index
