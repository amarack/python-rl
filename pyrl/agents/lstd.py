# Author: Will Dabney

from random import Random
import numpy

from rlglue.agent import AgentLoader as AgentLoader
from pyrl.rlglue.registry import register_agent

import pyrl.misc.matrix as matrix
import sarsa_lambda, qlearning
from pyrl.misc.parameter import *

@register_agent
class LSTD(sarsa_lambda.sarsa_lambda):
    """Least Squares Temporal Difference Learning (LSTD) agent.
    This is actually very nearly an implementation of LSTD-Q, as
    given in the paper:

    Least-Squares Policy Iteration. 2003.
    Michail Lagoudakis and Ronald Parr.

    The only difference is that we don't store the samples themselves,
    and instead just store A and b, meaning we can't reuse samples as
    effectively when the policy changes.
    """

    name = "Least Squares Temporal Difference Learning"

    @classmethod
    def agent_parameters(cls):
        param_set = super(LSTD, cls).agent_parameters()
        add_parameter(param_set, "lstd_update_freq", default=100, type=int, min=1, max=5000)
        remove_parameter(param_set, "alpha")
        return param_set

    def init_parameters(self):
        super(LSTD, self).init_parameters()
        self.lstd_gamma = self.gamma
        self.update_freq = int(self.params.setdefault('lstd_update_freq', 100))
        self.gamma = 1.0

    def init_stepsize(self, weights_shape, params):
        """Initializes the step-size variables, in this case meaning the A matrix and b vector.

        Args:
            weights_shape: Shape of the weights array
            params: Additional parameters.
        """
        # Using step_sizes for b
        # Using traces for z
        self.A = numpy.zeros((numpy.prod(weights_shape),numpy.prod(weights_shape)))
        self.step_sizes = numpy.zeros((numpy.prod(weights_shape),))
        self.lstd_counter = 0

    def shouldUpdate(self):
        self.lstd_counter += 1
        return self.lstd_counter % self.update_freq == 0

    def update(self, phi_t, phi_tp, reward):
        # A update...
        d = phi_t.flatten() - self.lstd_gamma * phi_tp.flatten()
        self.A = self.A + numpy.outer(self.traces.flatten(), d)
        self.step_sizes += self.traces.flatten() * reward

        if self.shouldUpdate():
            B = numpy.linalg.pinv(self.A)
            self.weights = numpy.dot(B, self.step_sizes).reshape(self.weights.shape)


@register_agent
class oLSTD(sarsa_lambda.sarsa_lambda):
    """Online Least Squares Temporal Difference Learning (oLSTD) agent.

    O(n^2) time complexity.

    """

    name = "Online Least Squares TD"

    def init_parameters(self):
        super(oLSTD, self).init_parameters()
        self.lstd_gamma = self.gamma
        self.gamma = 1.0

    @classmethod
    def agent_parameters(cls):
        param_set = super(oLSTD, cls).agent_parameters()
        remove_parameter(param_set, "alpha")
        return param_set

    def init_stepsize(self, weights_shape, params):
        """Initializes the step-size variables, in this case meaning the A matrix and b vector.

        Args:
            weights_shape: Shape of the weights array
            params: Additional parameters.
        """
        self.A = numpy.eye(numpy.prod(weights_shape))
        self.A += numpy.random.random(self.A.shape)*self.alpha
        self.step_sizes = numpy.zeros((numpy.prod(weights_shape),))
        self.lstd_counter = 0

    def update(self, phi_t, phi_tp, reward):
        d = phi_t.flatten() - self.lstd_gamma * phi_tp.flatten()
        self.step_sizes += self.traces.flatten() * reward

        self.A = matrix.SMInv(self.A, self.traces.flatten(), d, 1.)
        self.weights = numpy.dot(self.A, self.step_sizes).reshape(self.weights.shape)


@register_agent
class iLSTD(LSTD):
    """Incremental Least Squares Temporal Difference Learning (iLSTD) agent."""

    name = "Incremental Least Squares TD"

    def init_parameters(self):
        super(iLSTD, self).init_parameters()
        self.num_sweeps = int(self.params.setdefault('ilstd_sweeps', 1))

    @classmethod
    def agent_parameters(cls):
        param_set = super(iLSTD, cls).agent_parameters()
        add_parameter(param_set, "ilstd_sweeps", default=1, type=int, min=1, max=100)
        return param_set

    def update(self, phi_t, phi_tp, reward):
        #iLSTD
        # A update...
        d = numpy.outer(self.traces.flatten(), phi_t.flatten() - self.lstd_gamma*phi_tp.flatten())
        self.A = self.A + d
        self.step_sizes += self.traces.flatten() * reward - numpy.dot(d, self.weights.flatten())
        for i in range(self.num_sweeps):
            j = numpy.abs(self.step_sizes).argmax()
            self.weights.flat[j] += self.alpha * self.step_sizes[j]
            self.step_sizes -= self.alpha * self.step_sizes[j] * self.A.T[:,j]

@register_agent
class RLSTD(sarsa_lambda.sarsa_lambda):

    name = "Recursive Least Squares TD"

    def init_parameters(self):
        self.params.setdefault('alpha', 1.0)
        super(RLSTD, self).init_parameters()
        self.delta = self.params.setdefault('rlstd_delta', 1.0)

    @classmethod
    def agent_parameters(cls):
        param_set = super(RLSTD, cls).agent_parameters()
        add_parameter(param_set, "rlstd_delta", default=1, type=int, min=1, max=1000)
        return param_set

    def init_stepsize(self, weights_shape, params):
        self.A = numpy.eye(numpy.prod(weights_shape)) * self.delta

    def update(self, phi_t, phi_tp, reward):
        #RLS-TD(lambda)
        self.traces *= self.lmbda * self.gamma
        self.traces += phi_t

        # A update...
        d = numpy.dot(self.A, self.traces.flatten())
        K = d / (self.alpha + numpy.dot((phi_t - self.gamma * phi_tp).flatten(), d))
        self.A = matrix.SMInv(self.A, self.traces.flatten(), (phi_t - self.gamma*phi_tp).flatten(), self.alpha)
        self.weights += (reward - numpy.dot((phi_t - self.gamma * phi_tp).flatten(), self.weights.flatten())) * K.reshape(self.weights.shape)


@register_agent
class LSTDQ(qlearning.qlearning_agent):
    """Least Squares Temporal Difference Learning agent, LSTD-Q.
    This differs from LSTD class in that it holds onto the samples themselves
    and regenerates the matrix and vector (A, b) at each update based upon
    those samples and the current policy.

    From the paper:
    Least-Squares Policy Iteration. 2003.
    Michail Lagoudakis and Ronald Parr.
    """

    name = "LSTD-Q"

    @classmethod
    def agent_parameters(cls):
        param_set = super(LSTDQ, cls).agent_parameters()
        add_parameter(param_set, "lstd_num_samples", default=500, type=int, min=1, max=5000)
        add_parameter(param_set, "lstd_precond", default=0.1)
        remove_parameter(param_set, "alpha")
        return param_set

    def init_parameters(self):
        super(LSTDQ, self).init_parameters()
        self.lstd_gamma = self.gamma
        self.num_samples = int(self.params.setdefault('lstd_num_samples', 500))
        self.precond = self.params.setdefault('lstd_precond', 0.1)
        self.gamma = 1.0

    def init_stepsize(self, weights_shape, params):
        """Initializes the step-size variables, in this case meaning the A matrix and b vector.

        Args:
            weights_shape: Shape of the weights array
            params: Additional parameters.
        """
        # Data samples should hold num_samples, and each sample should
        # contain phi_t (|weights_shape|), state_t+1 (numStates), discState_t+1 (1), and reward_t (1)
        self.samples = numpy.zeros((self.num_samples, numpy.prod(weights_shape) + self.numStates + 2))
        self.lstd_counter = 0

    def shouldUpdate(self):
        self.lstd_counter += 1
        return self.lstd_counter % self.num_samples == 0

    def extractSample(self, sample):
        s = sample[:self.weights.size]
        state = sample[self.weights.size:self.weights.size+self.numStates]
        discState = sample[-2]
        qvalues = self.getActionValues(state, discState)
        a_p = self.getAction(state, discState)#values.argmax()
        s_p = numpy.zeros(self.weights.shape)
        s_p[discState, :, a_p] = self.basis.computeFeatures(state)
        return s, s_p.flatten(), sample[-1]

    def updateWeights(self):
        B = numpy.eye(self.weights.size) * self.precond
        b = numpy.zeros(self.weights.size)
        for sample in self.samples[:self.lstd_counter]:
            s, s_p, r = self.extractSample(sample)
            B = matrix.SMInv(B, s, (s - self.lstd_gamma * s_p), 1.0)
            b += s * r
        self.weights = numpy.dot(B, b).reshape(self.weights.shape)

    def update(self, phi_t, state, discState, reward):
        index = self.lstd_counter % self.num_samples
        self.samples[index, :phi_t.size] = phi_t.flatten()
        self.samples[index, phi_t.size:phi_t.size + self.numStates] = state.copy() if state is not None else numpy.zeros((self.numStates,))
        self.samples[index, -2] = discState
        self.samples[index, -1] = reward
        if self.shouldUpdate():
            self.updateWeights()

@register_agent
class LSPI(LSTDQ):
    """Least Squares Policy Iteration (LSPI) agent. Based around LSTDQ.

    From the paper:
    Least-Squares Policy Iteration. 2003.
    Michail Lagoudakis and Ronald Parr.
    """

    name = "LSPI"

    @classmethod
    def agent_parameters(cls):
        param_set = super(LSPI, cls).agent_parameters()
        add_parameter(param_set, "lspi_threshold", default=0.001)
        return param_set

    def init_parameters(self):
        super(LSPI, self).init_parameters()
        self.threshold = self.params.setdefault('lspi_threshold', 0.001) # Threshold for convergence

    def updateWeights(self):
        # Outer loop of LSPI algorithm, repeat until policy converges
        prev_weights = None
        while (prev_weights is None) or numpy.linalg.norm(prev_weights - self.weights.ravel()) >= self.threshold:
            prev_weights = self.weights.flatten()
            super(LSPI, self).updateWeights()


if __name__=="__main__":
    from pyrl.agents.skeleton_agent import runAgent
    runAgent(LSTD)

