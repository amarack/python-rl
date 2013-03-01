
import numpy
from sklearn import neighbors
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor

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

		# Initialize storage for training data
		self.experiences = numpy.zeros((params['max_experiences'], self.numActions, self.numContStates + 1))
		self.transitions = numpy.zeros((params['max_experiences'], self.numActions, self.numContStates + 1))
		self.terminates = numpy.zeros((params['max_experiences'],self.numActions))
		self.rewards = numpy.zeros((params['max_experiences'], self.numActions))

		self.exp_index = numpy.zeros((self.numActions,))
		self.has_fit = numpy.array([False]*self.numActions)

		# Set up model learning algorithm
		method = params.setdefault('method', 'knn')				
		# [reward_regressor, regressor for termination, classifier for disc states, regressor for each cont state]
		self.model = [[self._genregressor(method), self._genregressor(method), self._genclassifier(method)] + \
				      [self._genregressor(method) for i in range(self.numContStates)] for k in range(self.numActions)]

	def _genregressor(self, method):
		if method == "knn":
			return neighbors.KNeighborsRegressor(self.params['known_threshold'], weights=self.gaussianDist, warn_on_equidistant=False, algorithm="auto")
		elif method == "randforest":
			return RandomForestRegressor(n_jobs=2, n_estimators=5)
		else:
			return None

	def _genclassifier(self, method):
		if method == "knn":
			return neighbors.KNeighborsClassifier(self.params['known_threshold'], weights=self.gaussianDist, warn_on_equidistant=False, algorithm="auto")
		elif method == "randforest":
			return RandomForestClassifier(n_jobs=2, n_estimators=5)
		else:
			return None

	# Scales/Normalizes the state features to be in the interval [0,1]
	def normState(self, state):
		return (numpy.array(state) - self.feature_ranges[:,0]) / self.feature_span

	def denormState(self, state):
		return numpy.array(state)*self.feature_span + self.feature_ranges[:,0]

	def gaussianDist(self, dist):
		return numpy.exp(-(dist/(self.params['b'])**2))

	def isKnown(self, state, action):
		if not self.has_fit[action]:
			return False

		state = self.normState(state)
		method = self.params['method']
		if method == 'knn':
			dist, ind =self.model[action][0].kneighbors([state])
			n_sa = numpy.exp(-(dist/(self.params['b'])**2)).sum()
			return n_sa >= self.params['m']
		else:
			return False

	# list of list of states, first index is of action
	def areKnown(self, states):
		states = self.normState(states)
		method = self.params['method']
		if method == 'knn':
			# knn can simply get the distances to the neighbors, simple and easy
			known = []
			for a in range(self.numActions):
				if self.has_fit[a]:
					dist, ind = self.model[a][0].kneighbors(states[a])
					n_sa = numpy.exp(-(dist/(self.params['b'])**2)).sum(1)
					known += [(n_sa >= self.params['m'])]
				else:
					known += [numpy.array([False]*len(states[a]))]
			return known
		elif method == 'randforest':
			# random forests have no obvious way to get confidence intervals
			# instead we do the slow and dumb thing of getting the nearest neighbor 
			# and use that distance. 
			known = []
			for a in range(self.numActions):
				if self.has_fit[a]:
					split_predictions = map(lambda k: map(numpy.linalg.norm, k - self.experiences[:self.exp_index[a],a]), states[a])
					split_predictions = self.gaussianDist((numpy.array(split_predictions).min(1))**2)
					known += [split_predictions >= self.params['m']]
				else:
					known += [numpy.array([False]*len(states[a]))]

			return known
		else:
			return map(lambda k: (numpy.zeros((len(k),))!=0), states)

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

		
	def updateModel(self):
		if (self.exp_index >= self.params['update_freq']).all() and \
			    self.exp_index.sum() % self.params['update_freq'] == 0:
			for a in range(self.numActions):
				# update for action model a
				indices = numpy.where(self.terminates[:self.exp_index[a],a] == 0)

				# Reward model
				if self.params['importance_weight']:
					w = self.computeImpWeights(self.rewards[:self.exp_index[a],a])
					self.model[a][0].fit(self.experiences[:self.exp_index[a],a], self.rewards[:self.exp_index[a],a], sample_weight=w)
				else:
					self.model[a][0].fit(self.experiences[:self.exp_index[a],a], self.rewards[:self.exp_index[a],a])

				# Termination model
				if self.params['importance_weight']:
					w = self.computeImpWeights(self.terminates[:self.exp_index[a],a])
					self.model[a][1].fit(self.experiences[:self.exp_index[a],a], self.terminates[:self.exp_index[a],a], sample_weight=w)
				else:
					self.model[a][1].fit(self.experiences[:self.exp_index[a],a], self.terminates[:self.exp_index[a],a])

				# Discrete model
				self.model[a][2].fit(self.experiences[indices[0],a], self.transitions[indices[0],a,0])

				# Regression model
				for i in range(self.numContStates):
					self.model[a][i+3].fit(self.experiences[indices[0],a], self.transitions[indices[0],a,i+1])
				self.has_fit[a] = True

			return True
		else:
			return False

	def sampleStateActions(self, num_requested):
		sample = []
		ranges = self.getStateSpace()[0]
		for a in range(self.numActions):
			action_sample = numpy.random.uniform(low=self.feature_ranges[:,0], high=self.feature_ranges[:,1], 
							     size=(num_requested,len(self.feature_ranges)))
			sample += [action_sample.clip(min=self.feature_ranges[:,0], max=self.feature_ranges[:,1])]
		return sample

	def exploration_reward(self, state, known, rewards):
		method = self.params['method']
		if method == 'knn':
			rewards[numpy.invert(known)] = self.reward_range[1]
			return rewards
		elif method == 'randforest':
			rewards[numpy.invert(known)] = self.reward_range[1]
			return rewards

	def model_termination(self, pterm, known):
		method = self.params['method']
		if method == 'knn':
			pterm[numpy.invert(known)] = 1
			return pterm
		elif method == 'randforest':
			pterm[numpy.invert(known)] = 1
			return pterm
		
	# states should be a matrix formed from a list of lists
	# where the first list is over actions, and the second is a list 
	# of matrices of data for that action 
	def predictSet(self, states):
		pred = []
		known = self.areKnown(states)
		states = self.normState(states)
		for a in range(self.numActions):
			if self.has_fit[a]:
				predictions = numpy.array(map(lambda m: m.predict(states[a]), self.model[a])).T

				pState = predictions[:,2:]
				pTerminate = self.model_termination(predictions[:,1], known[a])
				pRewards = self.exploration_reward(states[a], known[a], predictions[:,0])
			
				ranges = self.getStateSpace()[0]
				if self.params['relative']:
					pred += [(self.denormState((pState + states[a]).clip(min=0, max=1)), pRewards, pTerminate)]
				else:
					pred += [(self.denormState(pState.clip(min=0, max=1)), pRewards, pTerminate)]
			else:
				pred += [([None]*len(states[a]), [None]*len(states[a]), [None]*len(states[a]))]
		return pred

	def predict(self, state, action):
		if not self.has_fit[action]:
			return None, None, None

		known = self.isKnown(state, action)
		state = self.normState(state)
		pState = numpy.zeros((self.numContStates+1,))

		predictions = map(lambda m: m.predict([state]), self.model[action])
		pState = numpy.array(predictions[2:]).flatten()
		pTerminate = self.model_termination(predictions[1], numpy.array([known]))
		pReward = self.exploration_reward(state, numpy.array([known]), predictions[0])
			
		ranges = self.getStateSpace()[0]
		# return full_state, reward, terminate
		if self.params['relative']:
			return self.denormState((state + pState).clip(min=0, max=1)), pReward, pTerminate
		else:
			return self.denormState(pState.clip(min=0, max=1)), pReward, pTerminate


        def updateExperience(self, lastState, action, newState, reward):
		if self.exp_index[action] >= self.params['max_experiences']:
			self.exp_index[action]+= 1
			return self.exp_index.sum() % self.params['update_freq'] == 0

		pnew = self.predict(lastState, action)
		lastState = self.normState(lastState)

		index = self.exp_index[action] % self.params['max_experiences']
		self.experiences[index,action, :] = lastState
		self.rewards[index, action] = reward
		if newState is not None:
			if pnew[0] is not None:
				print "#:P>", numpy.linalg.norm(newState - pnew[0])
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

        




