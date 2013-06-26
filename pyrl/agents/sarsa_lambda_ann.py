
from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.utils import TaskSpecVRLGLUE3
from pyrl.rlglue.registry import register_agent

from random import Random
import numpy
import copy
from skeleton_agent import *

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
class sarsa_lambda_ann(skeleton_agent):
    name = "Sarsa ANN"

    def init_parameters(self):
        self.epsilon = self.params.setdefault("epsilon", 0.01)
        self.lmbda = self.params.setdefault("lmbda", 0.7)
        self.gamma = self.params.setdefault("gamma", 1.0)
        self.net = None
        self.params = self.params
        self.softmax = self.params.setdefault("softmax", False)
        self.alpha = self.params.setdefault("alpha", 0.001)
        self.num_hidden = self.params.setdefault("num_hidden", 50)

    @classmethod
    def agent_parameters(cls):
        param_set = super(sarsa_lambda_ann, cls).agent_parameters()
        add_parameter(param_set, "alpha", default=0.01)
        add_parameter(param_set, "epsilon", default=0.1)
        add_parameter(param_set, "gamma", default=1.0)
        add_parameter(param_set, "lmbda", default=0.7)
        add_parameter(param_set, "num_hidden", default=10, type=int, min=1, max=500)

        # Parameters *NOT* used in parameter optimization
        add_parameter(param_set, "softmax", optimize=False, type=bool, default=False)
        return param_set

    def agent_init(self,taskSpec):
        self.init_parameters()
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
#        print delta
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

    def has_diverged(self):
        value = self.net.layers[0].np['w'].sum()
        return numpy.isnan(value) or numpy.isinf(value)


if __name__=="__main__":
    from pyrl.agents.skeleton_agent import runAgent
    runAgent(sarsa_lambda_ann)




