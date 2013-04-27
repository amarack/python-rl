
from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.utils import TaskSpecVRLGLUE3
from pyrl.rlglue.registry import register_agent

from random import Random
import numpy
import copy

import neurolab as nl

# Sarsa(lambda) agent which uses a feedforward neural network 
# for approximating the state-action value function. 
#
# Verified working, but a bit slow compared to linear func. approx.
# This implementation relies on neurolab for doing things that would be just 
# as easy to implement myself, but in the event that we want to try out 
# variations on the NN theme, neurolab has them implemented, tested and 
# working. 
@register_agent
class sarsa_lambda_ann(Agent):
	name = "Sarsa ANN"

	def __init__(self, epsilon=0.1, alpha=0.01, gamma=1.0, lmbda=0.7, softmax=False, num_hidden=10, params={}):
		self.randGenerator = Random()	
		self.lastAction=Action()
		self.lastObservation=Observation()

		self.epsilon = epsilon
		self.lmbda = lmbda
		self.gamma = gamma
		self.net = None
		self.params = params
		self.softmax = softmax
		self.alpha = float(alpha)
		self.num_hidden = num_hidden

	def agent_init(self,taskSpec):
		# Parse the task specification and set up the weights and such
		TaskSpec = TaskSpecVRLGLUE3.TaskSpecParser(taskSpec)
		if TaskSpec.valid:
			# Check observation form, and then set up number of features/states
			assert len(TaskSpec.getDoubleObservations())>0, "expecting at least one continuous observation"
			self.numStates=len(TaskSpec.getDoubleObservations())

			# Check action form, and then set number of actions
			assert len(TaskSpec.getIntActions())==1, "expecting 1-dimensional discrete actions"
			assert len(TaskSpec.getDoubleActions())==0, "expecting no continuous actions"
			assert not TaskSpec.isSpecial(TaskSpec.getIntActions()[0][0]), " expecting min action to be a number not a special value"
			assert not TaskSpec.isSpecial(TaskSpec.getIntActions()[0][1]), " expecting max action to be a number not a special value"
			self.numActions=TaskSpec.getIntActions()[0][1]+1;

			# Set up the function approximation
			self.net = nl.net.newff(TaskSpec.getDoubleObservations(), [self.num_hidden, self.numActions],[nl.net.trans.TanSig(), nl.net.trans.PureLin()])
			self.traces = copy.deepcopy(map(lambda x: x.np, self.net.layers))
			self.clearTraces()
		else:
			print "Task Spec could not be parsed: "+taskSpecString;

		self.lastAction=Action()
		self.lastObservation=Observation()

	def clearTraces(self):
		for layer in range(len(self.traces)):
			self.traces[layer]['b'][:] = 0.0
			self.traces[layer]['w'][:] = 0.0

	def decayTraces(self):
		for layer in range(len(self.traces)):
			self.traces[layer]['b'][:] *= self.lmbda * self.gamma
			self.traces[layer]['w'][:] *= self.lmbda * self.gamma

	def updateTraces(self, gradient, delta):
		for layer in range(len(self.traces)):
			self.traces[layer]['b'] += gradient[layer]['b']/delta
			self.traces[layer]['w'] += gradient[layer]['w']/delta

	def getAction(self, state):
		if self.softmax:
			return self.sample_softmax(state)
		else:
			return self.egreedy(state)

	def sample_softmax(self, state):
		Q = self.net.sim([state]).flatten()
		Q = numpy.exp(numpy.clip(Q/self.epsilon, -500, 500)) 
		Q /= Q.sum()
		
		# Would like to use numpy, but haven't upgraded enough (need 1.7)
		# numpy.random.choice(self.numActions, 1, p=Q)
		Q = Q.cumsum()
		return numpy.where(Q >= numpy.random.random())[0][0]
		
	def egreedy(self, state):
		if self.randGenerator.random() < self.epsilon:
			return self.randGenerator.randint(0,self.numActions-1)
		
		return self.net.sim([state]).argmax()
		
	def agent_start(self,observation):
		theState = numpy.array(observation.doubleArray)

		thisIntAction=self.getAction(theState)
		returnAction=Action()
		returnAction.intArray=[thisIntAction]

		# Clear traces
		self.clearTraces()

		self.lastAction = copy.deepcopy(returnAction)
		self.lastObservation = copy.deepcopy(observation)
		
		return returnAction
	
	def agent_step(self,reward, observation):
		newState = numpy.array(observation.doubleArray)
		lastState = numpy.array(self.lastObservation.doubleArray)
		lastAction = self.lastAction.intArray[0]

		newIntAction = self.getAction(newState)

		# Update eligibility traces
		self.decayTraces()
		self.update(lastState, lastAction, newState, newIntAction, reward)

		returnAction = Action()
		returnAction.intArray = [newIntAction]
		
		self.lastAction = copy.deepcopy(returnAction)
		self.lastObservation = copy.deepcopy(observation)
		return returnAction

	def update(self, x_t, a_t, x_tp, a_tp, reward):
		# Compute Delta (TD-error)
		Q_t = self.net.sim([x_t]).flatten()
		Q_tp = self.net.sim([x_tp])[0,a_tp] if x_tp is not None else 0.0
		deltaQ = Q_t.copy()
		delta = self.gamma*Q_tp + reward - Q_t[a_t]
		deltaQ[a_t] = self.gamma*Q_tp + reward

		grad = nl.tool.ff_grad_step(self.net, Q_t, deltaQ)
		self.updateTraces(grad, delta)

		# Update the weights 
		for layer in range(len(self.traces)):
			self.net.layers[layer].np['b'] -= self.alpha * delta * self.traces[layer]['b']
			self.net.layers[layer].np['w'] -= self.alpha * delta * self.traces[layer]['w']

		#newQ = self.net.sim([x_t]).flatten()
		#print Q_t[a_t], deltaQ[a_t], newQ[a_t]
	
	def agent_end(self,reward):
		lastState = self.lastObservation.doubleArray
		lastAction = self.lastAction.intArray[0]

		# Update eligibility traces
		self.decayTraces()
		self.update(lastState, lastAction, None, 0, reward)
	
	def agent_cleanup(self):
		pass
	
	def agent_message(self,inMessage):
		return name + " does not understand your message."

if __name__=="__main__":
	import argparse
	parser = argparse.ArgumentParser(description='Run SarsaLambda agent in network mode with linear function approximation.')
	parser.add_argument("--epsilon", type=float, default=0.1, help="Probability of exploration with epsilon-greedy.")
	parser.add_argument("--softmax", type=float, help="Use softmax policies with the argument giving tau, the divisor which scales values used when computing soft-max policies.")
	parser.add_argument("--stepsize", "--alpha", type=float, default=0.01, help="The step-size parameter which affects how far in the direction of the gradient parameters are updated.")
	parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor")
	parser.add_argument("--lambda", type=float, default=0.7, help="The eligibility traces decay rate. Set to 0 to disable eligibility traces.", dest='lmbda')
	parser.add_argument("--num_hidden", type=int, default=10, help="Number of hidden nodes to use in the Neural Network.")
	args = parser.parse_args()
	params = {}
	alpha = args.stepsize
	epsilon = args.epsilon
	softmax = False
	if args.softmax is not None:
		softmax = True
		epsilon = args.softmax

	AgentLoader.loadAgent(sarsa_lambda_ann(epsilon=epsilon, alpha=alpha, gamma=args.gamma, lmbda=args.lmbda, num_hidden=args.num_hidden, params=params, softmax=softmax))
