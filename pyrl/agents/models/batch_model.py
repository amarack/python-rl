
import numpy
from sklearn import neighbors

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

from model import ModelLearner

class BatchModel(ModelLearner):
	def __init__(self, numDiscStates, contStateRanges, numActions, rewardRange, params={}):
		ModelLearner.__init__(self, numDiscStates, contStateRanges, numActions, rewardRange, params)
		self.params.setdefault('relative', True)
		self.params.setdefault('update_freq', 20)
		self.params.setdefault('b', 2.0)
		self.params.setdefault('m', 5)
		self.experiences = numpy.zeros((params.setdefault('max_experiences', 5000), self.numActions, self.numContStates + 1))
		self.transitions = numpy.zeros((params['max_experiences'], self.numActions, self.numContStates + 1))
		self.terminates = numpy.zeros((params['max_experiences'],self.numActions))
		self.rewards = numpy.zeros((params['max_experiences'], self.numActions))
		self.exp_index = numpy.zeros((self.numActions,))
		self.has_fit = numpy.array([False]*self.numActions)
		method = params.setdefault('method', 'knn')
		self.params.setdefault('known_threshold', 200)
		if method == "knn":
			# [reward_regressor, regressor for termination, classifier for disc states, regressor for each cont state]
			self.model = [[neighbors.KNeighborsRegressor(self.params['known_threshold'], weights=self.gaussianDist), 
				      neighbors.KNeighborsRegressor(self.params['known_threshold'], weights=self.gaussianDist), 
				      neighbors.KNeighborsClassifier(self.params['known_threshold'], weights=self.gaussianDist)] + \
				      [neighbors.KNeighborsRegressor(self.params['known_threshold'], weights=self.gaussianDist) for i in range(self.numContStates)] \
					      for k in range(self.numActions)]
		else:
			self.model = None


	def gaussianDist(self, dist):
		return numpy.exp(-(dist/self.params['b'])**2)

	def isKnown(self, state, action):
		if not self.has_fit[action]:
			return False

		method = self.params['method']
		if method == 'knn':
			dist, ind =self.model[action][0].kneighbors([state])
			n_sa = numpy.exp(-(dist/self.params['b'])**2).sum()
			return n_sa >= self.params['m']
		else:
			return False

	# list of list of states, first index is of action
	def areKnown(self, states):
		method = self.params['method']
		if method == 'knn':
			known = []
			for a in range(self.numActions):
				if self.has_fit[a]:
					dist, ind = self.model[a][0].kneighbors(states[a])
					n_sa = numpy.exp(-(dist/self.params['b'])**2).sum()
					known += [(n_sa >= self.params['m'])]
				else:
					known += [numpy.array([False]*len(states[a]))]
			return known
		else:
			return map(lambda k: (numpy.zeros((len(k),))!=0), states)

	def updateModel(self):
		if (self.exp_index >= self.params['update_freq']).all() and \
			    self.exp_index.sum() % self.params['update_freq'] == 0:
			for a in range(self.numActions):
				# update for action model a
				indices = numpy.where(self.terminates[:self.exp_index[a],a] == 0)
				# Reward model
				self.model[a][0].fit(self.experiences[:self.exp_index[a],a], self.rewards[:self.exp_index[a],a])
				# Termination model
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
		for a in range(self.numActions):
			slice = self.experiences[:self.exp_index[a],a,:].copy()
			numpy.random.shuffle(slice)
			sample += [slice[:num_requested,:]]
		return sample

	# states should be a matrix formed from a list of lists
	# where the first list is over actions, and the second is a list 
	# of matrices of data for that action 
	def predictSet(self, states):
		pred = []
		known = self.areKnown(states)
		for a in range(self.numActions):
			predictions = numpy.array(map(lambda m: m.predict(states[a]), self.model[a])).T
			pRewards = predictions[:,0]
			pTerminate = predictions[:,1]
			pState = predictions[:,2:]
			pRewards[known[a]] = self.reward_range[1]
			if self.params['relative']:
				pred += [(pState + states[a], pRewards, pTerminate)]
			else:
				pred += [(pState, pRewards, pTerminate)]
		return pred

	def predict(self, state, action):
		vState = numpy.zeros((self.numContStates + 1,))
		vState[0] = state[0]
		vState[1:] = state[1:]

		pState = numpy.zeros((self.numContStates+1,))
		predictions = map(lambda m: m.predict([vState]), self.model[action])
		pReward, pTerminate = tuple(predictions[:2])
		if not self.isKnown(vState, action):
			pReward = self.reward_range[1]

		pState[:] = predictions[2:]

		# return full_state, reward, terminate
		if self.params['relative']:
			return vState + pState, pReward, pTerminate
		else:
			return pState, pReward, pTerminate


        def updateExperience(self, lastState, action, newState, reward):
		index = self.exp_index[action] % self.params['max_experiences']
		self.experiences[index,action, 0] = lastState[0]
		self.experiences[index,action, 1:] = lastState[1]
		self.rewards[index, action] = reward

		if newState is not None:
			if self.params['relative']:
				self.transitions[index, action, 0] = newState[0] - lastState[0]
				self.transitions[index, action, 1:] = newState[1] - lastState[1]
			else:
				self.transitions[index, action, 0] = newState[0]
				self.transitions[index, action, 1:] = newState[1]

			self.terminates[index, action] = 0
		else:
			self.transitions[index,action, 0] = 0
			self.transitions[index,action, 1:] = 0
			self.terminates[index, action] = 1

		self.exp_index[action] += 1
		return self.updateModel()

        




