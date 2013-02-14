

class Planner:

	def __init__(self, model, params={}):
		self.model = model

        def updateExperience(self, lastState, newState, reward):
		if self.model.updateExperience(lastState, newState, reward):
			self.updatePlan()

        def updatePlan(self):
		pass
	
	def getAction(self, state):
		pass

        

