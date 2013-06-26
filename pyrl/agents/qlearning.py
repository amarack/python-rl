
# Author: Will Dabney

from random import Random
import numpy
import copy

from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.utils import TaskSpecVRLGLUE3
from pyrl.rlglue.registry import register_agent

import sarsa_lambda
import stepsizes

@register_agent
class qlearning_agent(sarsa_lambda.sarsa_lambda):
    name = "Q-Learning"

    def agent_step(self,reward, observation):
        """Take one step in an episode for the agent, as the result of taking the last action.

        Args:
            reward: The reward received for taking the last action from the previous state.
            observation: The next observation of the episode, which is the consequence of taking the previous action.

        Returns:
            The next action the RL agent chooses to take, represented as an RLGlue Action object.
        """

        newState = numpy.array(list(observation.doubleArray))
        lastState = numpy.array(list(self.lastObservation.doubleArray))
        lastAction = self.lastAction.intArray[0]

        newDiscState = self.getDiscState(observation.intArray)
        lastDiscState = self.getDiscState(self.lastObservation.intArray)

        # Update eligibility traces
        phi_t = numpy.zeros(self.traces.shape)
        phi_t[lastDiscState, :, lastAction] = self.basis.computeFeatures(lastState)

        self.update_traces(phi_t, None)
        self.update(phi_t, newState, newDiscState, reward)

        # QLearning can choose action after update
        newIntAction = self.getAction(newState, newDiscState)
        returnAction=Action()
        returnAction.intArray=[newIntAction]

        self.lastAction=copy.deepcopy(returnAction)
        self.lastObservation=copy.deepcopy(observation)
        return returnAction

    def getActionValues(self, state, discState):
        if state is not None:
            return numpy.dot(self.weights[discState,:,:].T, self.basis.computeFeatures(state))
        else:
            return numpy.zeros((self.numActions,))

    def update(self, phi_t, state, discState, reward):
        qvalues = self.getActionValues(state, discState)
        a_tp = qvalues.argmax()
        phi_tp = numpy.zeros(self.traces.shape)
        if state is not None:
            phi_tp[discState, :, a_tp] = self.basis.computeFeatures(state)

        # Compute Delta (TD-error)
        delta = self.gamma*qvalues[a_tp] + reward - numpy.dot(self.weights.flatten(), phi_t.flatten())

        # Update the weights with both a scalar and vector stepsize used
        # (Maybe we should actually make them both work together naturally)
        self.weights += self.rescale_update(phi_t, phi_tp, delta, reward, delta*self.traces)

    def agent_end(self,reward):
        """Receive the final reward in an episode, also signaling the end of the episode.

        Args:
            reward: The reward received for taking the last action from the previous state.
        """

        lastState = numpy.array(list(self.lastObservation.doubleArray))
        lastAction = self.lastAction.intArray[0]

        lastDiscState = self.getDiscState(self.lastObservation.intArray)

        # Update eligibility traces
        phi_t = numpy.zeros(self.traces.shape)
        phi_t[lastDiscState, :, lastAction] = self.basis.computeFeatures(lastState)

        self.update_traces(phi_t, None)
        self.update(phi_t, None, 0, reward)



if __name__=="__main__":
    from pyrl.agents.skeleton_agent import runAgent
    runAgent(qlearning_agent)





