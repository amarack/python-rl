
# A collection of Policy Gradient reinforcement learning
# algorithms.

# Author: Will Dabney

import numpy, copy

from rlglue.types import Action
from rlglue.types import Observation

from pyrl.rlglue.registry import register_agent
from pyrl.misc.matrix import vector_angle, SMInv
import sarsa_lambda
import stepsizes
from pyrl.misc.parameter import *

class policy_gradient(sarsa_lambda.sarsa_lambda):
    def agent_start(self,observation):
        self.step_count = 0
        return sarsa_lambda.sarsa_lambda.agent_start(self, observation)

    def update(self, phi_t, phi_tp, reward, compatFeatures):
        pass

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
        newIntAction = self.getAction(newState, newDiscState)

        phi_t = numpy.zeros((self.weights.shape[0], self.weights.shape[1]))
        phi_tp = numpy.zeros((self.weights.shape[0], self.weights.shape[1]))
        phi_t[lastDiscState, :] = self.basis.computeFeatures(lastState)
        phi_tp[newDiscState, :] = self.basis.computeFeatures(newState)

        self.step_count += 1
        self.update(phi_t, phi_tp, reward, self.getCompatibleFeatures(phi_t, lastAction, reward, phi_tp, newIntAction))

        returnAction=Action()
        returnAction.intArray=[newIntAction]
        self.lastAction=copy.deepcopy(returnAction)
        self.lastObservation=copy.deepcopy(observation)
        return returnAction

    def agent_end(self,reward):
        """Receive the final reward in an episode, also signaling the end of the episode.

        Args:
            reward: The reward received for taking the last action from the previous state.
        """
        lastState = numpy.array(list(self.lastObservation.doubleArray))
        lastAction = self.lastAction.intArray[0]
        lastDiscState = self.getDiscState(self.lastObservation.intArray)

        phi_t = numpy.zeros((self.weights.shape[0], self.weights.shape[1]))
        phi_tp = numpy.zeros((self.weights.shape[0], self.weights.shape[1]))
        phi_t[lastDiscState, :] = self.basis.computeFeatures(lastState)

        self.update(phi_t, phi_tp, reward, self.getCompatibleFeatures(phi_t, lastAction, reward, phi_tp, 0))

    def getAction(self, state, discState):
        features = numpy.zeros(self.weights.shape[:-1])
        features[discState, :] = self.basis.computeFeatures(state)
        policy = self.getPolicy(features)
        return numpy.where(policy.cumsum() >= numpy.random.random())[0][0]

    def getPolicy(self, features):
        if self.softmax:
            return self.softmax_policy(features)
        else:
            return self.gauss_policy(features)

    def gauss_policy(self, features):
        # Not currently supported...
        return self.softmax_policy(features)

    def softmax_policy(self, features):
        # Compute softmax policy
        policy = numpy.dot(self.weights.reshape((features.size,self.numActions)).T, features.ravel())
        policy = numpy.exp(numpy.clip(policy/self.epsilon, -500, 500))
        policy /= policy.sum()
        return policy

    def getCompatibleFeatures(self, features, action, reward, next_features, next_action):
        policy = -1.0 * self.getPolicy(features)
        policy[action] += 1.0
        features = numpy.repeat(features.reshape((features.size,1)), self.numActions, axis=1)
        return numpy.dot(features, numpy.diag(policy)) # This is probably a slow way to do it


@register_agent
class REINFORCE(policy_gradient):
    """REINFORCE policy gradient algorithm.

    From the paper:
    Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning,
    Ronald Williams, 1992.
    """

    name = "REINFORCE"

    def agent_init(self,taskSpec):
        super(REINFORCE, self).agent_init(self,taskSpec)
        self.baseline_numerator = numpy.zeros(self.weights.shape)
        self.baseline_denom = numpy.zeros(self.weights.shape)
        self.gradient_estimate = numpy.zeros(self.weights.shape)
        self.ep_count = 0

    def init_parameters(self):
        super(REINFORCE, self).init_parameters(self)
        self.num_rollouts = self.params.setdefault('num_rollouts', 5)

    @classmethod
    def agent_parameters(cls):
        param_set = super(REINFORCE, cls).agent_parameters()
        add_parameter(param_set, "num_rollouts", default=5, type=int, min=1, max=50)
        return param_set

    def agent_start(self,observation):
        if self.ep_count > 0:
            self.baseline_numerator += (self.traces**2) * self.Return
            self.baseline_denom += self.traces**2
            self.gradient_estimate += self.traces * -((self.baseline_numerator/self.baseline_denom) - self.Return)
        if self.ep_count == self.num_rollouts:
            # update the parameters...
            self.weights += self.step_sizes * self.gradient_estimate
            # Clear estimates for next set of roll outs
            self.gradient_estimate.fill(0.0)
            self.baseline_numerator.fill(0.0)
            self.baseline_denom.fill(0.0)
            self.ep_count = 0

        self.ep_count += 1
        self.Return = 0.0
        return super(REINFORCE, self).agent_start(self, observation)

    def update(self, phi_t, phi_tp, reward, compatFeatures):
        self.traces += compatFeatures
        self.Return += (self.gamma**(self.step_count-1.)) * reward


@register_agent
class twotime_ac(policy_gradient):
    """Two-Timescale Actor-Critic algorithm.

    This is an implementation of Algorithm 1 from the paper:
    Natural Actor-Critic Algorithms,
    Shalabh Bhatnagar, Richard S. Sutton, Mohammad Ghavamzadeh, and Mark Lee, 2009.
    """

    name = "Two-Timescale Actor-Critic"

    def agent_init(self,taskSpec):
        policy_gradient.agent_init(self, taskSpec)
        self.v = numpy.zeros((self.weights.shape[0], self.weights.shape[1]))

    def agent_start(self,observation):
        self.avg_reward = None
        return policy_gradient.agent_start(self, observation)

    def init_parameters(self):
        policy_gradient.init_parameters(self)
        self.beta = self.params.setdefault("beta", 0.001)

    @classmethod
    def agent_parameters(cls):
        param_set = super(twotime_ac, cls).agent_parameters()
        add_parameter(param_set, "beta", default=0.001)
        return param_set

    def update(self, phi_t, phi_tp, reward, compatFeatures):
        if self.avg_reward is None:
            self.avg_reward = reward
        else:
            self.avg_reward = (1. - self.lmbda) * self.avg_reward + self.alpha * reward
        # Compute Delta (TD-error)
        delta = numpy.dot(self.v.flatten(), (self.gamma * phi_tp - phi_t).flatten()) + reward - self.avg_reward
        self.v += self.beta * self.lmbda * delta * phi_t
        # Update the weights with both a scalar and vector stepsize used
        self.weights += self.step_sizes * delta * compatFeatures

@register_agent
class twotime_nac(twotime_ac):
    """Two-Timescale Natural Actor-Critic algorithm.

    This is an implementation of Algorithm 3 from the paper:
    Natural Actor-Critic Algorithms,
    Shalabh Bhatnagar, Richard S. Sutton, Mohammad Ghavamzadeh, and Mark Lee, 2009.
    """

    name = "Two-Timescale Natural Actor-Critic"
    def update(self, phi_t, phi_tp, reward, compatFeatures):
        if self.avg_reward is None:
            self.avg_reward = reward
        else:
            self.avg_reward = (1. - self.lmbda) * self.avg_reward + self.alpha * reward
        # Compute Delta (TD-error)
        delta = numpy.dot(self.v.flatten(), (self.gamma * phi_tp - phi_t).flatten()) + reward - self.avg_reward
        self.v += self.beta * self.lmbda * delta * phi_t
        self.traces = numpy.dot(numpy.eye(compatFeatures.size) - self.beta * self.lmbda * numpy.outer(compatFeatures, compatFeatures), self.traces.flatten()).reshape(self.traces.shape)
        self.traces += self.beta * self.lmbda * delta * compatFeatures
        # Update the weights with both a scalar and vector stepsize used
        self.weights += self.step_sizes * self.traces


@register_agent
class nac_lstd(policy_gradient):
    """Natural Actor-Critic with LSTD-Q algorithm.

    From the paper:
    Natural Actor-Critic,
    Jan Peters and Stefan Schaal, 2007.

    This deviates from the pseudo-code given in the paper because it uses the Sheman-Morrison
    formula to do incremental updates to the matrix inverse.
    """

    name = "Natural Actor-Critic with LSTD-Q"
    def init_parameters(self):
        policy_gradient.init_parameters(self)
        self.nac_freq = self.params.setdefault("nac_freq", 200)
        self.nac_precond = self.params.setdefault('precond', 0.01)

    @classmethod
    def agent_parameters(cls):
        param_set = super(nac_lstd, cls).agent_parameters()
        add_parameter(param_set, "nac_freq", default=200, type=int, min=1, max=1000)
        add_parameter(param_set, "precond", default=0.01)
        return param_set

    def agent_init(self,taskSpec):
        sarsa_lambda.sarsa_lambda.agent_init(self, taskSpec)
        self.traces = numpy.zeros((numpy.prod(self.weights.shape[:-1]) + self.weights.size,))
        self.A = numpy.eye(self.traces.size)*self.nac_precond
#        self.A += numpy.random.random(self.A.shape)*self.nac_precond
        self.b = numpy.zeros((self.traces.size,))

    def appendBaseline(self, phi_t, phi_tp, compatFeatures):
        phi_tilde = numpy.zeros(self.traces.shape)
        phi_hat = numpy.zeros(self.traces.shape)
        phi_tilde[:phi_tp.size] = phi_tp.flatten()
        phi_hat[:phi_t.size] = phi_t.flatten()
        phi_hat[phi_t.size:] = compatFeatures.flatten()
        return phi_hat, phi_tilde

    def update(self, phi_t, phi_tp, reward, compatFeatures):
        phi_hat, phi_tilde = self.appendBaseline(phi_t, phi_tp, compatFeatures)

        self.traces *= self.lmbda
        self.traces += phi_hat
        self.A = SMInv(self.A, self.traces, phi_hat - self.gamma * phi_tilde, 1.)
        self.b += self.traces * reward
        if self.step_count % self.nac_freq == 0:
            parameters = numpy.dot(self.A, self.b)
            # Update the weights with both a scalar and vector stepsize used
            self.weights += self.step_sizes * parameters[phi_t.size:].reshape(self.weights.shape)

@register_agent
class nac_sarsa(policy_gradient):
    """Natural Actor-Critic with SARSA(lambda).

    While fundamentally the same as twotime_nac (Algorithm 3 of BSGL's paper),
    this implements NACS which uses a different form of the same update equations.
    The main difference is in this algorithm's avoidance of the average reward
    accumulator.

    From the paper:
    Natural Actor-Critic using Sarsa(lambda),
    Philip S. Thomas, 2012.
    """

    name = "Natural Actor-Critic with Sarsa"

    def has_diverged(self):
        value = self.advantage_weights.sum() + self.value_weights.sum()
        return numpy.isnan(value) or numpy.isinf(value) or super(nac_sarsa, self).has_diverged()

    def init_parameters(self):
        policy_gradient.init_parameters(self)
        self.beta = self.params.setdefault("beta", 0.001)
        self.nac_freq = self.params.setdefault("nac_freq", 200)

    @classmethod
    def agent_parameters(cls):
        param_set = super(nac_sarsa, cls).agent_parameters()
        add_parameter(param_set, "nac_freq", default=200, type=int, min=1, max=1000)
        add_parameter(param_set, "beta", default=0.001)
        return param_set

    def agent_init(self,taskSpec):
        sarsa_lambda.sarsa_lambda.agent_init(self, taskSpec)
        self.traces = numpy.zeros((numpy.prod(self.weights.shape[:-1]) + self.weights.size,)) # combined e_t^w and e_t^v
        self.value_weights = numpy.zeros((numpy.prod(self.weights.shape[:-1]),))
        self.advantage_weights = numpy.zeros((self.weights.size,))

    def update(self, phi_t, phi_tp, reward, compatFeatures):
        phi_hat = numpy.zeros(self.traces.shape)
        phi_hat[:phi_t.size] = phi_t.flatten()
        phi_hat[phi_t.size:] = compatFeatures.flatten()

        self.traces *= self.lmbda
        self.traces += phi_hat

        delta = numpy.dot(self.value_weights, (self.gamma * phi_tp - phi_t).flatten()) + reward
        self.advantage_weights += self.beta * (delta - numpy.dot(self.advantage_weights, compatFeatures.flatten())) * self.traces[self.value_weights.size:]
        self.value_weights += self.beta * delta * self.traces[:self.value_weights.size]

        if self.step_count % self.nac_freq == 0:
            # Update the weights with both a scalar and vector stepsize used
            self.weights += self.step_sizes * self.advantage_weights.reshape(self.weights.shape) / numpy.linalg.norm(self.advantage_weights)


if __name__=="__main__":
    from pyrl.agents.skeleton_agent import runAgent
    runAgent(nac_sarsa)





