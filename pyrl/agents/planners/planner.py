
from random import Random

class Planner:

	def __init__(self, model, **kwargs):
		self.model = model
		self.gamma = kwargs.setdefault('gamma', 1.0)
		self.params = kwargs
		self.randGenerator = Random()
		
		
	def planner_init(self, numDiscStates, contFeatureRanges, numActions, rewardRange):
		pass

	def randomize_parameters(self, **args):
		"""Generate parameters randomly, constrained by given named parameters.

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
		self.params['gamma'] = args.setdefault('gamma', numpy.random.random())
		return [self.params['gamma']]

        def updateExperience(self, lastState, action, newState, reward):
		if self.model.updateExperience(lastState, action, newState, reward):
			self.updatePlan()

        def updatePlan(self):
		pass
	
	def getAction(self, state):
		pass

        

