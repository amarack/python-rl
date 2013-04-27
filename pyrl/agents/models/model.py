import numpy

class ModelLearner:

	def __init__(self, **kwargs):
		self.params = kwargs

	def model_init(self, numDiscStates, contFeatureRanges, numActions, rewardRange):
		self.numDiscStates = numDiscStates
		self.numContStates = len(contFeatureRanges)
		self.numActions = numActions
		self.reward_range = rewardRange
		self.feature_ranges = numpy.array([[0, self.numDiscStates-1]] + list(contFeatureRanges))
		self.feature_span = numpy.ones((len(self.feature_ranges),))
		non_constants = self.feature_ranges[:,0]!=self.feature_ranges[:,1]
		self.feature_span[non_constants] = self.feature_ranges[non_constants,1] - self.feature_ranges[non_constants,0]

	def randomize_parameters(self, **args):
		"""Generate parameters randomly, constrained by given named parameters.

		If used, this must be called before agent_init in order to have desired effect.
		
		Parameters that fundamentally change the algorithm are not randomized over. For 
		example, basis and softmax fundamentally change the domain and have very few values 
		to be considered. They are not randomized over.

		Basis parameters, on the other hand, have many possible values and ARE randomized. 

		Args:
			**args: Named parameters to fix, which will not be randomly generated

		Returns:
			List of resulting parameters of the class. Will always be in the same order. 
			Empty list if parameter free.

		"""
		return []

        def updateExperience(self, lastState, action, newState, reward):
		return False
	
        def getStateSpace(self):
		return self.feature_ranges, self.numActions

	# This method does not gaurantee that num_requested is filled, but will not 
	# provide more than num_requested.
	def sampleStateActions(self, num_requested):
		pass

	def predict(self, state, action):
		pass

	def predictSet(self, states):
		pass


	def isKnown(self, state, action):
		return False


