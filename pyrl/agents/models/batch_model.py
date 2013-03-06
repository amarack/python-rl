
import numpy
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVR, SVC, OneClassSVM
from sklearn.gaussian_process import GaussianProcess

# Batch methods:
#     knn
#     SVC & SVR
#     DecisionTreeClassifier and DecisionTreeRegressor 
#     Random Forests: ExtraTreesRegressor/Classifier
#     Gaussian Processes
#
# These correspond to the structure learning algorithms:
#     Fitted-Rmax
#     (Unknown)
#     SLF-Rmax
#     TEXPLORE
#     (Don't know what it is called, but know people have done this)


# TODO: Need to refactor things so that we can switch between exploration models, or types. 
# right now we are only doing Rmax, but the other one to try immediately is add to the expected reward b * variance for predicitons
# * another cool idea is to use the OneClassSVM to come up with values for the known/unknown method. 

from model import ModelLearner

class BatchModel(ModelLearner):
	def __init__(self, numDiscStates, contStateRanges, numActions, rewardRange, params={}):
		ModelLearner.__init__(self, numDiscStates, contStateRanges, numActions, rewardRange, params)
		# Set up parameter defaults
		self.params.setdefault('relative', True)
		self.params.setdefault('update_freq', 20)
		self.params.setdefault('b', 2.0)
		self.params.setdefault('known_threshold', 1)
		self.params.setdefault('m', 0.95)
		self.params.setdefault('max_experiences', 700)
		self.params.setdefault('importance_weight', False)
		self.params.setdefault('known_method', 'nndist')
		#self.params.setdefault('known_method', 'oneclass')

		# Initialize storage for training data
		self.experiences = numpy.zeros((params['max_experiences'], self.numActions, self.numContStates + 1))
		self.transitions = numpy.zeros((params['max_experiences'], self.numActions, self.numContStates + 1))
		self.terminates = numpy.zeros((params['max_experiences'],self.numActions))
		self.rewards = numpy.zeros((params['max_experiences'], self.numActions))

		self.exp_index = numpy.zeros((self.numActions+1,))
		self.has_fit = numpy.array([False]*self.numActions)
		self.predConst = numpy.zeros((self.numActions, self.numContStates+3)).tolist()
		self.sigma_threshold = numpy.zeros((self.numActions, self.numContStates))
		if self.params['known_method'] == 'oneclass':
			self.density_estimator = [OneClassSVM(nu=0.05, kernel="rbf", gamma=0.1) for a in range(self.numActions)]

		# Set up model learning algorithm
		method = params.setdefault('method', 'knn')				
		# [reward_regressor, regressor for termination, classifier for disc states, regressor for each cont state]
		self.model = [[self._genregressor(method, True), self._genregressor(method, True), self._genclassifier(method)] + \
				      [self._genregressor(method) for i in range(self.numContStates)] for k in range(self.numActions)]

	def _genregressor(self, method, isClassLike=False):
		if method == "knn":
			return neighbors.KNeighborsRegressor(self.params['known_threshold'], weights=self.gaussianDist, warn_on_equidistant=False, algorithm="auto")
		elif method == "randforest":
			return RandomForestRegressor(n_jobs=2, n_estimators=5)
		elif method == "svm":
			return SVR(C=1.0, epsilon=0.00001, tol=1e-6)
		elif method == "gp":
			if isClassLike:
				return GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1., random_start=100, nugget=1.0e-8, corr='linear')
			else:
				return GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1., random_start=100, nugget=1.0e-10)
#			return GaussianProcess(theta0=0.1, thetaL=.0001, thetaU=1., random_start=100, nugget=1.0e-10, corr='cubic')
		else:
			return None

	def _genclassifier(self, method):
		if method == "knn":
			return neighbors.KNeighborsClassifier(self.params['known_threshold'], weights=self.gaussianDist, warn_on_equidistant=False, algorithm="auto")
		elif method == "randforest":
			return RandomForestClassifier(n_jobs=2, n_estimators=5)
		elif method == "svm":
			return SVC()
		elif method == "gp":
			return GaussianProcess(theta0=0.1, thetaL=.001, thetaU=1., nugget=0.5)
		else:
			return None

	# Scales/Normalizes the state features to be in the interval [0,1]
	def normState(self, state):
		return (numpy.array(state) - self.feature_ranges[:,0]) / self.feature_span

	def denormState(self, state):
		return numpy.array(state)*self.feature_span + self.feature_ranges[:,0]

	def normStates(self, states):
		return map(self.normState, states)

	def denormStates(self, states):
		return map(self.denormState, states)

	def gaussianDist(self, dist):
		return numpy.exp(-(dist/(self.params['b'])**2))

	def isKnown(self, state, action):
		if not self.has_fit[action]:
			return False

		state = self.normState(state)
		method = self.params['method']
		if method == 'knn':
			dist, ind =self.model[action][-1].kneighbors([state])
			n_sa = numpy.exp(-(dist/(self.params['b'])**2)).sum()
			return n_sa >= self.params['m']
		elif method == 'gp':
			predictions = numpy.array(map(lambda d: d.predict(state, eval_MSE=True), self.model[action][3:]))
			return (numpy.sqrt(predictions[:,1]) <= self.sigma_threshold[action]).all()
			#return numpy.sqrt(predictions[:,1]).mean() <= self.params['m']
		else:
			method = self.params['known_method']
			if method == 'nndist':
				# random forests have no obvious way to get confidence intervals
				# instead we do the slow and dumb thing of getting the nearest neighbor 
				# and use that distance. 
				if self.has_fit[action]:
					split_predictions = map(numpy.linalg.norm, state - self.experiences[:self.exp_index[action],action])
					split_predictions = self.gaussianDist((numpy.array(split_predictions).min()))
					return split_predictions >= self.params['m']
				else:
					return False
			elif method == 'oneclass':
				if self.has_fit[action]:
					return (self.density_estimator[action].predict([state]) > 0).all()
				else:
					return False

			else:
				return False


	# list of list of states, first index is of action
	def areKnown(self, states):
		states = self.normStates(states)
		method = self.params['method']
		if method == 'knn':
			# knn can simply get the distances to the neighbors, simple and easy
			known = []
			for a in range(self.numActions):
				if self.has_fit[a]:
					dist, ind = self.model[a][-1].kneighbors(states[a])
					n_sa = numpy.exp(-(dist/(self.params['b'])**2)).sum(1)
					known += [(n_sa >= self.params['m'])]
				else:
					known += [numpy.array([False]*len(states[a]))]
			return known
		elif method == 'gp':
			known = []
			for a in range(self.numActions):
				if self.has_fit[a]:
					predictions = numpy.array(map(lambda d: d.predict(states[a], eval_MSE=True), self.model[a][3:]))
					#known += [numpy.sqrt(predictions[:,1]).mean(0) <= self.params['m']]
					known += [map(lambda k: (k <= self.sigma_threshold[a]).all(), numpy.sqrt(predictions[:,1]).T)]
				else:
					known += [numpy.array([False]*len(states[a]))]
			return known
		else:
			method = self.params['known_method']
			if method == 'nndist':
				# random forests have no obvious way to get confidence intervals
				# instead we do the slow and dumb thing of getting the nearest neighbor 
				# and use that distance. 
				known = []
				for a in range(self.numActions):
					if self.has_fit[a]:
						split_predictions = map(lambda k: map(numpy.linalg.norm, k - self.experiences[:self.exp_index[a],a]), states[a])
						split_predictions = self.gaussianDist((numpy.array(split_predictions).min(1)))
						known += [split_predictions >= self.params['m']]
						import csv
						with open('dd.dat', 'wb') as f:
							csvw = csv.writer(f)
							csvw.writerows(map(lambda k: list(k[0]) + [k[1]] + list(self.predict(k[0], a)), zip(self.denormState(states[a]), split_predictions)))
					else:
						known += [numpy.array([False]*len(states[a]))]

				return known
			elif method == 'oneclass':
				known = []
				for a in range(self.numActions):
					if self.has_fit[a]:
						known += [self.density_estimator[a].predict(states[a]) > 0]
					else:
						known += [numpy.array([False]*len(states[a]))]
				return known
			else:
				return map(lambda k: (numpy.zeros((len(k),))!=0), states)

	def getKnown(self, states):
		states = self.normStates(states)
		method = self.params['method']
		if method == 'knn':
			# knn can simply get the distances to the neighbors, simple and easy
			known = []
			for a in range(self.numActions):
				if self.has_fit[a]:
					dist, ind = self.model[a][-1].kneighbors(states[a])
					n_sa = numpy.exp(-(dist/(self.params['b'])**2)).sum(1)
					known += [zip(n_sa, self.denormState(self.experiences[ind,a]))]
				else:
					known += [numpy.array([0]*len(states[a]))]
			return known
		elif method == 'gp':
			known = []
			for a in range(self.numActions):
				if self.has_fit[a]:
					predictions = numpy.array(map(lambda d: d.predict(states[a], eval_MSE=True), self.model[a][3:]))
					if (predictions[:,1] >  1.0).any():
						print "## ERROR: ", predictions.tolist()
					known += [numpy.sqrt(predictions[:,1]).mean(0)]
				else:
					known += [numpy.array([0]*len(states[a]))]
			return known
		else:
			method = self.params['known_method']
			if method == 'nndist':
				# random forests have no obvious way to get confidence intervals
				# instead we do the slow and dumb thing of getting the nearest neighbor 
				# and use that distance. 
				known = []
				for a in range(self.numActions):
					if self.has_fit[a]:
						split_predictions = map(lambda k: map(numpy.linalg.norm, k - self.experiences[:self.exp_index[a],a]), states[a])
						indx = numpy.array(split_predictions).argmin(1)
						split_predictions = self.gaussianDist((numpy.array(split_predictions).min(1)))
						known += [zip(split_predictions, self.denormState(self.experiences[indx,a]))]
					else:
						known += [numpy.array([0]*len(states[a]))]

				return known
			elif method == 'oneclass':
				known = []
				for a in range(self.numActions):
					if self.has_fit[a]:
						known += [self.density_estimator[a].predict(states[a])]
					else:
						known += [numpy.array([0]*len(states[a]))]
				return known
			else:
				return map(lambda k: numpy.zeros((len(k),)), states)

	# Compute importance weights for a data set, higher weight for rarer values
	def computeImpWeights(self, data):
		hist, bin_edges = numpy.histogram(data)
		hist = numpy.array(hist, dtype=float)
		nonzero = numpy.where(hist > 0)
		hist[nonzero] = 1.0/hist[nonzero]
		bins = zip(bin_edges[:-1], bin_edges[1:])
		data_weights = numpy.zeros((len(data),))
		for bin, weight in zip(bins[:-1], hist):
			indices = numpy.where((data >= bin[0]) & (data < bin[1]))
			data_weights[indices] = weight
		indices = numpy.where((data >= bins[-1][0]) & (data <= bins[-1][1]))
		data_weights[indices] = hist[-1]
		return data_weights


	def buildConf(self, model, X, Y, zscore=1.960):
		r = numpy.random.normal(scale=0.01, size=X.shape)
		y_pred, y_mse = model.predict(X+r, eval_MSE=True)
		return zscore * numpy.sqrt(y_mse).max()

	def fitFactorModel(self, model, X, Y, allow_iw=True):
		if len(numpy.unique(Y)) > 1:
			if self.params['importance_weight']:
				w = self.computeImpWeights(Y)
				model.fit(X, Y, sample_weight=w*len(w))
			else:
				try:
					model.fit(X, Y)
				except:
					print X
					print Y
					import sys
					sys.exit()
				
			return None
		else:
			return Y[0]

	def updateModel(self):
		if (self.exp_index[:-1] >= self.params['update_freq']).all() and \
			    self.exp_index.sum() % self.params['update_freq'] == 0:
			for a in range(self.numActions):
				# update for action model a
				indices = numpy.where(self.terminates[:self.exp_index[a],a] == 0)

				# Reward model
				self.predConst[a][0] = self.fitFactorModel(self.model[a][0], 
									   self.experiences[:self.exp_index[a],a], 
									   self.rewards[:self.exp_index[a],a], True)

				# Termination model
				self.predConst[a][1] = self.fitFactorModel(self.model[a][1], 
									self.experiences[:self.exp_index[a],a], 
									self.terminates[:self.exp_index[a],a], True)

				# Discrete model
				self.predConst[a][2] = self.fitFactorModel(self.model[a][2], 
									   self.experiences[indices[0],a], 
									   self.transitions[indices[0],a,0])

				# Regression model
				for i in range(self.numContStates):
					self.predConst[a][i+3] = self.fitFactorModel(self.model[a][i+3], 
										    self.experiences[indices[0],a], 
										    self.transitions[indices[0],a,i+1])
				if self.params['method'] == 'gp':
					for i in range(self.numContStates):
						self.sigma_threshold[a][i] = self.buildConf(self.model[a][i+3], 
											    self.experiences[indices[0],a], 
											    self.transitions[indices[0],a,i+1])
				if self.params['known_method'] == 'oneclass':
					self.density_estimator[a].fit(self.experiences[:self.exp_index[a],a])
				self.has_fit[a] = True
			#print "#()", ','.join(map(str, self.sigma_threshold.flatten()))
			return True
		else:
			return False

	def sampleStateActions(self, num_requested):
		sample = []
		ranges = self.getStateSpace()[0]
		for a in range(self.numActions):
			rnd = range(int(min(self.exp_index[a], self.experiences.shape[0])))
			numpy.random.shuffle(rnd)
			action_sample = numpy.random.uniform(low=self.feature_ranges[:,0], high=self.feature_ranges[:,1], 
							     size=(num_requested,len(self.feature_ranges)))
			action_sample[:20] = self.denormState(self.experiences[rnd[:20],a])
			sample += [action_sample.clip(min=self.feature_ranges[:,0], max=self.feature_ranges[:,1])]
		return sample

	def exploration_reward(self, state, known, rewards):
		method = self.params['known_method']
		if method == 'nndist' or method == 'oneclass':
			rewards[numpy.invert(known)] = self.reward_range[1]
			return rewards
		else:
			return rewards

	def model_termination(self, pterm, known):
		method = self.params['known_method']
		if method == 'nndist' or method == 'oneclass':
			pterm[numpy.invert(known)] = 1
			return pterm
		else:
			return pterm
		

	# states should be a matrix formed from a list of lists
	# where the first list is over actions, and the second is a list 
	# of matrices of data for that action 
	def predictSet(self, states):
		pred = []
		known = self.areKnown(states)
		states = self.normStates(states)
		for a in range(self.numActions):
			if self.has_fit[a]:
				predictions = numpy.array(map(lambda (m,p): m.predict(states[a]) if p is None else numpy.ones((len(states[a]),))*p, 
							      zip(self.model[a], self.predConst[a]))).T

				pState = predictions[:,2:]
				pTerminate = self.model_termination(predictions[:,1], known[a])
				pRewards = self.exploration_reward(states[a], known[a], predictions[:,0])
			
				ranges = self.getStateSpace()[0]
				if self.params['relative']:
					pred += [(self.denormState((pState + states[a]).clip(min=0, max=1)), pRewards, pTerminate.clip(min=0, max=1))]
				else:
					pred += [(self.denormState(pState.clip(min=0, max=1)), pRewards, pTerminate.clip(min=0, max=1))]
			else:
				pred += [([None]*len(states[a]), [None]*len(states[a]), [None]*len(states[a]))]
		return pred

	def predict(self, state, action):
		if not self.has_fit[action]:
			return None, None, None

		known = self.isKnown(state, action)
		state = self.normState(state)
		pState = numpy.zeros((self.numContStates+1,))

		#predictions = map(lambda m: m.predict([state]), self.model[action])
		predictions = map(lambda (m,p): m.predict([state]) if p is None else [p], zip(self.model[action], self.predConst[action]))
		pState = numpy.array(predictions[2:]).flatten()
		pTerminate = self.model_termination(numpy.array(predictions[1]), numpy.array([known]))
		pReward = self.exploration_reward(state, numpy.array([known]), numpy.array(predictions[0]))
			
		ranges = self.getStateSpace()[0]
		# return full_state, reward, terminate
		if self.params['relative']:
			return self.denormState((state + pState).clip(min=0, max=1)), pReward, pTerminate.clip(min=0, max=1)#max(0, min(1, pTerminate))
		else:
			return self.denormState(pState.clip(min=0, max=1)), pReward, pTerminate.clip(min=0, max=1)#max(0, min(1, pTerminate))


        def updateExperience(self, lastState, action, newState, reward):
		if self.exp_index[action] >= self.params['max_experiences']:
			self.exp_index[action]+= 1
			return self.exp_index.sum() % self.params['update_freq'] == 0

		pnew = self.predict(lastState, action)
		lastState = self.normState(lastState)
		if self.exp_index[action] > 0:
			dist = numpy.array(map(numpy.linalg.norm, self.experiences[:self.exp_index[action],action] - numpy.array(lastState)))
			if dist.min() <= 1.0e-12:
				print "# IGNORING>....", lastState, self.experiences[dist.argmin(),action], dist.min()
				self.exp_index[-1] += 1
				return self.exp_index.sum() % self.params['update_freq'] == 0
				
		index = self.exp_index[action] % self.params['max_experiences']
		self.experiences[index,action, :] = lastState
		self.rewards[index, action] = reward
		if newState is not None:
			if pnew[0] is not None:
				print "#:P>", numpy.linalg.norm(newState - pnew[0]), newState, pnew[0]

			newState = self.normState(newState)
			if self.params['relative']:
				self.transitions[index, action, :] = newState - lastState
			else:
				self.transitions[index, action, :] = newState

			self.terminates[index, action] = 0
		else:
			self.transitions[index,action, 0] = 0
			self.transitions[index,action, 1:] = 0
			self.terminates[index, action] = 1
			
		self.exp_index[action] += 1
		return self.updateModel()

        




