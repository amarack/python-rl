
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

class BatchModel:
	def __init__(self, numDiscStates, numContStates, numActions, params={}):
		self.numDiscStates = numDiscStates
		self.numContStates = numContStates
		self.numActions = numActions
		self.params = params
		self.params.setdefault('relative', True)
		self.params.setdefault('update_freq', 10)
		self.experiences = numpy.zeros((params.setdefault('max_experiences', 1000), numContStates + 1))
		self.transitions = numpy.zeros((params['max_experiences'], numContStates + 1))
		self.terminates = numpy.zeros((params['max_experiences'],))
		self.rewards = numpy.zeros((params['max_experiences'],))
		self.exp_index = 0

		method = params.setdefault('method', 'knn')
		if method == "knn":
			# [reward_regressor, regressor for termination, classifier for disc states, regressor for each cont state]
			self.model = [neighbors.KNeighborsRegressor(self.params.setdefault('known_threshold', 1), weights='distance')]
			self.model.append(neighbors.KNeighborsRegressor(self.params.setdefault('known_threshold', 1), weights='distance'))
			self.model.append(neighbors.KNeighborsClassifier(self.params.setdefault('known_threshold', 1), weights='distance'))
			for i in range(numContStates):
				self.model.append(neighbors.KNeighborsRegressor(self.params.setdefault('known_threshold', 1), weights='distance'))
		else:
			self.model = None

	def updateModel(self):
		if self.exp_index % self.params['update_freq'] == 0:
			# update
			indices = numpy.where(self.terminates[:self.exp_index] == 0)
			self.model[0].fit(self.experiences[indices], self.rewards[indices[0]])
			self.model[1].fit(self.experiences[:self.exp_index], self.terminates[:self.exp_index])
			if self.numDiscStates > 1:
				self.model[2].fit(self.experiences[indices], self.transitions[indices[0],0])
			for i in range(self.numContStates):
				self.model[i+3].fit(self.experiences[indices], self.transitions[indices[0],i+1])
			return True
		else:
			return False

	def predict(self, state):
		vState = numpy.zeros((self.numContStates + 1,))
		vState[0] = state[0]
		vState[1:] = state[1:]
		pState = numpy.zeros((self.numContStates+1,))
		pReward = self.model[0].predict([vState])
		pTerminate = self.model[1].predict([vState])

		if self.numDiscStates > 1:
			pState[0] = self.model[2].predict([vState])

		for i in range(self.numContStates):
			pState[i+1] = self.model[i+3].predict([vState])

		# return full_state, reward, terminate
		if self.params['relative']:
			return vState + pState, pReward, pTerminate
		else:
			return pState, pReward, pTerminate


        def updateExperience(self, lastState, newState, reward):
		index = self.exp_index % self.params['max_experiences']
		self.experiences[index,0] = lastState[0]
		self.experiences[index,1:] = lastState[1]
		self.rewards[index] = reward

		if newState is not None:
			if self.params['relative']:
				self.transitions[index,0] = newState[0] - lastState[0]
				self.transitions[index,1:] = newState[1] - lastState[1]
			else:
				self.transitions[index,0] = newState[0]
				self.transitions[index,1:] = newState[1]

			self.terminates[index] = 0
		else:
			self.transitions[index,0] = 0
			self.transitions[index,1:] = 0
			self.terminates[index] = 1

		self.exp_index += 1
		return self.updateModel()

        




