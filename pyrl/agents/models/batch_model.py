
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
		self.params.setdefault('known_threshold', 1)
		self.params.setdefault('m', 0.95*self.params['known_threshold']) 
		self.experiences = numpy.zeros((params.setdefault('max_experiences', 700), self.numActions, self.numContStates + 1))
		self.transitions = numpy.zeros((params['max_experiences'], self.numActions, self.numContStates + 1))
		self.terminates = numpy.zeros((params['max_experiences'],self.numActions))
		self.rewards = numpy.zeros((params['max_experiences'], self.numActions))
		self.exp_index = numpy.zeros((self.numActions,))
		self.has_fit = numpy.array([False]*self.numActions)
		method = params.setdefault('method', 'knn')
		if method == "knn":
			# [reward_regressor, regressor for termination, classifier for disc states, regressor for each cont state]
			al = 'auto'
			self.model = [[neighbors.KNeighborsRegressor(self.params['known_threshold'], weights=self.gaussianDist, warn_on_equidistant=False, algorithm=al), 
				      neighbors.KNeighborsRegressor(self.params['known_threshold'], weights=self.gaussianDist, warn_on_equidistant=False, algorithm=al), 
				      neighbors.KNeighborsClassifier(self.params['known_threshold'], weights=self.gaussianDist, warn_on_equidistant=False, algorithm=al)] + \
				      [neighbors.KNeighborsRegressor(self.params['known_threshold'], weights=self.gaussianDist, warn_on_equidistant=False, algorithm=al) for i in range(self.numContStates)] \
					      for k in range(self.numActions)]
		else:
			self.model = None


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
			known = []
			for a in range(self.numActions):
				if self.has_fit[a]:
					dist, ind = self.model[a][0].kneighbors(states[a])
					n_sa = numpy.exp(-(dist/(self.params['b'])**2)).sum(1)
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
		ranges = self.getStateSpace()[0]
		for a in range(self.numActions):
			sample += [numpy.random.uniform(low=self.feature_ranges[:,0], high=self.feature_ranges[:,1], 
							size=(num_requested,len(self.feature_ranges))).clip(min=self.feature_ranges[:,0], max=self.feature_ranges[:,1])]
		return sample

	# states should be a matrix formed from a list of lists
	# where the first list is over actions, and the second is a list 
	# of matrices of data for that action 
	def predictSet(self, states):
		pred = []
		known = self.areKnown(states)
		states = self.normState(states)
		for a in range(self.numActions):
			predictions = numpy.array(map(lambda m: m.predict(states[a]), self.model[a])).T
			pRewards = predictions[:,0]
			pTerminate = predictions[:,1]
			pState = predictions[:,2:]
			pRewards[numpy.invert(known[a])] = self.reward_range[1]
			pTerminate[numpy.invert(known[a])] = 1
			ranges = self.getStateSpace()[0]
			if self.params['relative']:
				pred += [(self.denormState((pState + states[a]).clip(min=0, max=1)), pRewards, pTerminate)]
			else:
				pred += [(self.denormState(pState.clip(min=0, max=1)), pRewards, pTerminate)]
		return pred

	def predict(self, state, action):
		state = self.normState(state)
		pState = numpy.zeros((self.numContStates+1,))
		predictions = map(lambda m: m.predict([state]), self.model[action])
		pReward, pTerminate = tuple(predictions[:2])
		if not self.isKnown(state, action):
			pReward = self.reward_range[1]
			pTerminate = 1

		pState[:] = predictions[2:]
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

		lastState = self.normState(lastState)

		index = self.exp_index[action] % self.params['max_experiences']
		self.experiences[index,action, :] = lastState
		self.rewards[index, action] = reward
		if newState is not None:
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

        




