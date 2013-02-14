

class Planner:

	def __init__(self, model, params={}):
		self.model = model
		self.params = params

        def updateExperience(self, lastState, action, newState, reward):
		if self.model.updateExperience(lastState, action, newState, reward):
			self.updatePlan()

        def updatePlan(self):
		pass
	
	def getAction(self, state):
		pass

        

