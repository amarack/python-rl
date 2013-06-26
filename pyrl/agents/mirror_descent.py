
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

import sarsa_lambda, qlearning
import stepsizes
from pyrl.misc.parameter import *

def pnorm_linkfunc(weights, order):
    """Link function induced by the p-norm."""
    if (weights==0.0).all():
        return numpy.zeros(weights.shape)
    return numpy.sign(weights) * numpy.abs(weights)**(order-1) / numpy.linalg.norm(weights, ord=order)**(order - 2)


@register_agent
class md_qlearn(qlearning.qlearning_agent):
    """Sparse Mirror Descent Q-Learning using a p-norm distance generating function.
    Set the sparsity parameter to zero to get Mirror Descent Q-Learning.
    From the paper:

    Sparse Q-Learning with Mirror Descent,
    Sridhar Mahadevan and Bo Liu, 2012.
    """
    name = "Sparse Mirror Descent Q-Learning"

    def init_parameters(self):
        qlearning.qlearning_agent.init_parameters(self)
        self.sparsity = self.params.setdefault('sparsity', 0.01)

    @classmethod
    def agent_parameters(cls):
        param_set = super(md_qlearn, cls).agent_parameters()
        add_parameter(param_set, "sparsity", default=0.01)
        return param_set

    def agent_start(self,observation):
        returnAction = qlearning.qlearning_agent.agent_start(self, observation)
        self.pnorm = 2. * max(1, numpy.log10(numpy.prod(self.weights.shape)))
        self.qnorm = self.pnorm / (self.pnorm - 1.)
        return returnAction

    def proj_dual(self, weights):
        return pnorm_linkfunc(weights.flatten(), self.qnorm).reshape(weights.shape)

    def proj_primal(self, weights):
        return pnorm_linkfunc(weights.flatten(), self.pnorm).reshape(weights.shape)

    def update(self, phi_t, state, discState, reward):
        qvalues = self.getActionValues(state, discState)
        a_tp = qvalues.argmax()

        # Compute Delta (TD-error)
        delta = self.gamma*qvalues[a_tp] + reward - numpy.dot(self.weights.flatten(), phi_t.flatten())

        # Update dual weights
        dual_weights = self.proj_dual(self.weights)
        dual_weights += self.step_sizes * delta * self.traces

        # Truncate weights for sparsity
        dual_weights = numpy.sign(dual_weights) * (numpy.abs(dual_weights) - self.step_sizes * self.sparsity).clip(0.0)

        # Update the weights
        self.weights = self.proj_primal(dual_weights)


@register_agent
class md_sarsa(sarsa_lambda.sarsa_lambda):
    """Sparse Mirror Descent Q-Learning using a p-norm distance generating function.
    Set the sparsity parameter to zero to get Mirror Descent Q-Learning.
    From the paper:

    Sparse Q-Learning with Mirror Descent,
    Sridhar Mahadevan and Bo Liu, 2012.
    """
    name = "Sparse Mirror Descent Sarsa"

    def init_parameters(self):
        sarsa_lambda.sarsa_lambda.init_parameters(self)
        self.sparsity = self.params.setdefault('sparsity', 0.01)

    @classmethod
    def agent_parameters(cls):
        param_set = super(md_sarsa, cls).agent_parameters()
        add_parameter(param_set, "sparsity", default=0.01)
        return param_set

    def agent_start(self,observation):
        returnAction = sarsa_lambda.sarsa_lambda.agent_start(self, observation)
        self.pnorm = 2. * max(1, numpy.log10(numpy.prod(self.weights.shape)))
        self.qnorm = self.pnorm / (self.pnorm - 1.)
        return returnAction

    def proj_dual(self, weights):
        return pnorm_linkfunc(weights.flatten(), self.qnorm).reshape(weights.shape)

    def proj_primal(self, weights):
        return pnorm_linkfunc(weights.flatten(), self.pnorm).reshape(weights.shape)

    def update(self, phi_t, phi_tp, reward):
        # Compute Delta (TD-error)
        delta = numpy.dot(self.weights.flatten(), (self.gamma * phi_tp - phi_t).flatten()) + reward

        # Update dual weights
        dual_weights = self.proj_dual(self.weights)
        dual_weights += self.step_sizes * delta * self.traces

        # Truncate weights for sparsity
        dual_weights = numpy.sign(dual_weights) * (numpy.abs(dual_weights) - self.step_sizes * self.sparsity).clip(0.0)

        # Update the weights
        self.weights = self.proj_primal(dual_weights)


# NOTE: This agent is not working at all. Not sure yet what is wrong
@register_agent
class cmd_qlearn(md_qlearn):
    """Sparse Mirror Descent Q-Learning using a p-norm distance generating function.
    Set the sparsity parameter to zero to get Mirror Descent Q-Learning.
    From the paper:

    Sparse Q-Learning with Mirror Descent,
    Sridhar Mahadevan and Bo Liu, 2012.
    """
    name = "Composite Mirror Descent Q-Learning"

    def init_parameters(self):
        super(cmd_qlearn,self).init_parameters()
        self.covariance = None

    def update(self, phi_t, state, discState, reward):
        qvalues = self.getActionValues(state, discState)
        a_tp = qvalues.argmax()

        # Compute Delta (TD-error)
        delta = self.gamma*qvalues[a_tp] + reward - numpy.dot(self.weights.flatten(), phi_t.flatten())

        if self.covariance is None:
            self.covariance = numpy.zeros(phi_t.shape)

        self.covariance += phi_t**2
        H = numpy.sqrt(self.covariance)
        H[H == 0.0] = 1.0
        nobis_term = self.step_sizes / H

        # Update the weights
        update = self.weights - nobis_term * delta * self.traces
        self.weights = numpy.sign(update) * (numpy.abs(update) - nobis_term * self.sparsity)


@register_agent
class mdba_qlearn(md_qlearn):
    """Sparse Mirror Descent Q-Learning with Non-Linear Basis Adaptation,
    using a p-norm distance generating function. Set the sparsity parameter
    to zero to get Mirror Descent Q-Learning.
    From the paper:

    Basis Adaptation for Sparse Nonlinear Reinforcement Learning
    Sridhar Mahadevan, Stephen Giguere, and Nicholas Jacek, 2013.
    """
    name = "Sparse Mirror Descent Q-Learning with Non-Linear Basis Adaptation"

    def init_parameters(self):
        self.params['basis'] = 'fourier' # Force to use fourier basis for adaptation
        md_qlearn.init_parameters(self)
        self.beta = self.params.setdefault('nonlinear_lr', 1.e-6)

    @classmethod
    def agent_parameters(cls):
        param_set = super(mdba_qlearn, cls).agent_parameters()
        add_parameter(param_set, "nonlinear_lr", default=1.e-6)
        add_parameter(param_set, "basis", optimize=False, type=str, choices=['fourier'], default='fourier')
        return param_set


    def agent_init(self, taskSpec):
        md_qlearn.agent_init(self, taskSpec)
        self.freq_scale = numpy.ones((self.weights.shape[1],))

    def basisGradient(self, state):
        # Derivative of basis features w.r.t. frequency scale
        sfeatures = numpy.dot(numpy.dot(numpy.diag(1./self.freq_scale), self.basis.multipliers), numpy.array([self.basis.scale(state[i],i) for i in range(len(state))]))
        return -numpy.pi * sfeatures * numpy.sin(numpy.pi * self.freq_scale * sfeatures)

    def update(self, phi_t, state, discState, reward):
        qvalues = self.getActionValues(state, discState)
        a_tp = qvalues.argmax()

        if state is not None:
            lastState = numpy.array(list(self.lastObservation.doubleArray))
            lastAction = self.lastAction.intArray[0]
            lastDiscState = self.getDiscState(self.lastObservation.intArray)
            update_fs = numpy.zeros(self.freq_scale.shape)
            deltaGrad = numpy.zeros(self.freq_scale.shape)

            # Compute Delta (smoothed TD-error)
            delta = self.gamma*qvalues[a_tp] + reward - numpy.dot(self.weights.flatten(), phi_t.flatten())
            # logSumExp is equiv to log( sum[ exp(qvalues) ] )
            logSumExp = qvalues[a_tp] + numpy.log(numpy.exp(qvalues - qvalues[a_tp]).sum())
            # approxMaxGrad is equiv to deriv of smoothed max(Q) w.r.t. Q[a]
            approxMaxGrad = numpy.exp(qvalues - logSumExp)

            # Compute gradient of smoothed TD error
            fa_grad = self.basisGradient(state)
            for a in range(self.numActions):
                deltaGrad += approxMaxGrad[a] * (fa_grad * self.weights[discState,:,a])
            fa_grad = self.basisGradient(lastState)
            deltaGrad = self.gamma * deltaGrad - (fa_grad * self.weights[lastDiscState, :,lastAction])

            # Compute the update to the basis scale features
            update_fs = self.beta * delta * deltaGrad
            # Do MDA update for weights
            md_qlearn.update(self, phi_t, state, discState, reward)

            # Update frequency scaling
            update_fs += self.freq_scale
            # Change scaling on multipliers
            self.basis.multipliers = numpy.dot(numpy.diag(update_fs/self.freq_scale), self.basis.multipliers)
            self.freq_scale = update_fs
        else:
            md_qlearn.update(self, phi_t, state, discState, reward)


if __name__=="__main__":
    from pyrl.agents.skeleton_agent import runAgent
    runAgent(mdba_qlearn)




