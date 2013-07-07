#################################################################
# Step-Size algorithms for Reinforcement Learning Agents        #
#                                                               #
# These algorithms are for setting the step-size at each        #
# time step for a value function based reinforcement learning   #
# agent. Generally they are written for the sarsa and qlearning #
# agents in PyRL, and thus assume the existence of the RL       #
# parameters (gamma, lmbda, alpha) and make use of them as      #
# needed.                                                       #
#                                                               #
# Author: Will Dabney                                           #
#################################################################

import numpy, scipy.linalg
from pyrl.misc import matrix
from pyrl.rlglue.registry import register_agent
import argparse
from pyrl.misc.parameter import *

def genAdaptiveAgent(stepsize_class, agent_class):
    """Generate an RL agent by combining an existing agent with a step-size algorithm."""

    @register_agent
    class AdaptiveAgent(stepsize_class, agent_class):
        name = "Adaptive (" + stepsize_class.name + ") " + agent_class.name
        def __init__(self, **args):
            agent_class.__init__(self, **args)

        @classmethod
        def agent_parameters(cls):
            return argparse.ArgumentParser(parents=[agent_class.agent_parameters(), stepsize_class.agent_parameters()])

    return AdaptiveAgent

class AdaptiveStepSize(object):
    name = "Fixed StepSize"

    def init_stepsize(self, weights_shape, params):
        self.step_sizes = numpy.ones(weights_shape) * self.alpha

    def rescale_update(self, phi_t, phi_tp, delta, reward, descent_direction):
        return self.step_sizes * descent_direction

    @classmethod
    def agent_parameters(cls):
        """Produces an argparse.ArgumentParser for all the parameters of this RL agent
        algorithm. Specifically, parameters mean to be optimized (e.g. in a parameter search)
        should be added to the argument group 'optimizable'. The best way to do this is with
        the functions contained in pyrl/misc/parameter.py. Specifically, parameter_set for
        creating a new set of parameters, and add_parameter to add parameters (use optimize=False)
        to indicate that the parameter should not be optimized over.
        """
        return parameter_set(cls.name, description="Parameters required for running an RL agent algorithm.")



class GHS(AdaptiveStepSize):
    """Generalized Harmonic Stepsize algorithm for scalar step-sizes.

    Follows the equation: a_t = a_0 * (a / a + t - 1),
    for parameters a_0 and a
    """
    name = "GHS"
    def init_stepsize(self, weights_shape, params):
        self.step_sizes = numpy.ones(weights_shape) * self.alpha
        self.last_update = numpy.zeros(weights_shape)
        self.ghs_param = params.setdefault('ghs_a', 10.0)
        self.ghs_counter = 1

    def rescale_update(self, phi_t, phi_tp, delta, reward, descent_direction):
        self.step_sizes.fill(self.alpha * self.ghs_param/(self.ghs_param + self.ghs_counter - 1))
        self.ghs_counter += 1
        return self.step_sizes * descent_direction

    @classmethod
    def agent_parameters(cls):
        param_set = super(GHS, cls).agent_parameters()
        add_parameter(param_set, "ghs_a", default=10., min=1., max=10000.)
        return param_set

class McClains(AdaptiveStepSize):
    """McClain's formula for scalar step-size

    Follows the equation: a_t = a_{t-1} / (1 + a_{t-1} - a)
    unless t = 0, then use a_0, for parameters a_0 and a
    """
    name = "McClains"
    def init_stepsize(self, weights_shape, params):
        if self.alpha < params.setdefault('mcclain_a', 0.01):
            a = self.alpha
            self.alpha = params.setdefault('mcclain_a', 0.01)
            params['mcclain_a'] = a

        self.step_sizes = numpy.ones(weights_shape) * self.alpha
        self.mcclain_param = params.setdefault('mcclain_a', 0.01)

    def rescale_update(self, phi_t, phi_tp, delta, reward, descent_direction):
        self.step_sizes.fill(self.alpha)
        self.alpha /= (1 + self.alpha - self.mcclain_param)
        return self.step_sizes * descent_direction

    @classmethod
    def agent_parameters(cls):
        param_set = super(McClains, cls).agent_parameters()
        add_parameter(param_set, "mcclain_a", default=0.01)
        return param_set

class STC(AdaptiveStepSize):
    """Search-Then-Converge formula for scalar step-size

    Follows the equation: a_t = a_{t-1} * (1 + (c/a_0) * (t/N)) / (1 + (c/a_0) * (t/N) + N * (t^2/N^2))
    for parameters a_0 the initial stepsize, c the target stepsize, N the pivot point.
    N (the pivot point) is simply approximately how many steps at which the formula begins to
    converge more rather than search more.
    """
    name = "STC"
    def init_stepsize(self, weights_shape, params):
        self.step_sizes = numpy.ones(weights_shape) * self.alpha
        self.stc_a0 = self.alpha
        self.stc_c = params.setdefault('stc_c', 1000000.0)
        self.stc_N = params.setdefault('stc_N', 500000.0)
        self.stc_counter = 0

    def rescale_update(self, phi_t, phi_tp, delta, reward, descent_direction):
        self.alpha *= (1 + (self.stc_c * self.stc_counter)/(self.stc_a0 * self.stc_N))
        self.alpha /= (1 + (self.stc_c * self.stc_counter)/(self.stc_a0 * self.stc_N) + self.stc_N*(self.stc_counter**2)/self.stc_N**2)
        self.step_sizes.fill(self.alpha)
        self.stc_counter += 1
        return self.step_sizes * descent_direction

    @classmethod
    def agent_parameters(cls):
        param_set = super(STC, cls).agent_parameters()
        add_parameter(param_set, "stc_c", default=1000000.0, min=1., max=1.e10)
        add_parameter(param_set, "stc_c", default=500000.0, min=1., max=1.e6)
        return param_set


class RProp(AdaptiveStepSize):
    """RProp algorithm for vector step-sizes.

    From the paper:
    Riedmiller, M. and Braun, H. (1993).
    A direct adaptive method for faster backpropagation learning: The RPROP algorithm.
    """
    name = "RProp"
    def init_stepsize(self, weights_shape, params):
        self.step_sizes = numpy.ones(weights_shape) * self.alpha
        self.last_update = numpy.zeros(weights_shape)
        self.eta_low = params.setdefault('rprop_eta_low', 0.01)
        self.eta_high = params.setdefault('rprop_eta_high', 1.2)

    def rescale_update(self, phi_t, phi_tp, delta, reward, descent_direction):
        sign_changes = numpy.where(self.last_update * delta * self.traces <= 0)
        self.step_sizes.fill(self.eta_high)
        self.step_sizes[sign_changes] = self.eta_low
        return self.step_sizes * descent_direction

    @classmethod
    def agent_parameters(cls):
        param_set = super(RProp, cls).agent_parameters()
        add_parameter(param_set, "rprop_eta_high", default=0.01)
        add_parameter(param_set, "rprop_eta_low", default=1.2, min=0.5, max=2.)
        return param_set


class Autostep(AdaptiveStepSize):
    """Autostep algorithm for vector step-sizes.

    From the paper:
    Mahmood, A. R., Sutton, R. S., Degris, T., and Pilarski, P. M. 2012.
    Tuning-free step-size adaptation.
    """
    name = "Autostep"
    def init_stepsize(self, weights_shape, params):
        self.step_sizes = numpy.ones(weights_shape) * self.alpha
        self.h = numpy.zeros((numpy.prod(weights_shape),))
        self.v = numpy.zeros((numpy.prod(weights_shape),))
        # Autostep should not be used with eligibility traces (in current form)
        #self.lmbda = 0.0
        self.mu = params.setdefault('autostep_mu', 1.0e-2)
        self.tau = params.setdefault('autostep_tau', 1.0e4)

    def rescale_update(self, phi_t, phi_tp, delta, reward, descent_direction):
        x = self.traces.flatten()
        deltaTerm = delta * x * self.h
        alphas = self.step_sizes.flatten()
        self.v = numpy.max([numpy.abs(deltaTerm),
                            self.v + (1.0/self.tau)*alphas*(x**2)*(numpy.abs(deltaTerm) - self.v)],0)
        v_not_zero = self.v != 0.0
        alphas[v_not_zero] = alphas[v_not_zero] * numpy.exp(self.mu * deltaTerm[v_not_zero]/self.v[v_not_zero])
        M = numpy.max([numpy.dot(alphas, x**2), 1.0])
        self.step_sizes = (alphas / M).reshape(self.step_sizes.shape)
        plus_note = ( 1.0 - self.step_sizes.flatten() * x**2 )
        # This may or may not be used depending on which paper you read
        #plus_note[plus_note < 0] = 0.0
        self.h = self.h * plus_note + self.step_sizes.flatten()*delta*x
        return self.step_sizes * descent_direction

    @classmethod
    def agent_parameters(cls):
        param_set = super(Autostep, cls).agent_parameters()
        add_parameter(param_set, "autostep_mu", default=1.e-2)
        add_parameter(param_set, "autostep_tau", default=1.e4, min=1., max=1.e6)
        return param_set


class AlphaBounds(AdaptiveStepSize):
    """AlphaBounds adaptive scalar step-size.

    From the paper:
    Dabney, W. and A. G. Barto (2012).
    Adaptive Step-Size for Online Temporal Difference Learning.
    """
    name = "AlphaBound"
    def init_stepsize(self, weights_shape, params):
        self.alpha = 1.0
        self.step_sizes = numpy.ones(weights_shape) * self.alpha

    def rescale_update(self, phi_t, phi_tp, delta, reward, descent_direction):
        deltaPhi = (self.gamma * phi_tp - phi_t).flatten()
        denomTerm = numpy.dot(self.traces.flatten(), deltaPhi.flatten())
        self.alpha = numpy.min([self.alpha, 1.0/numpy.abs(denomTerm)])
        self.step_sizes.fill(self.alpha)
        return self.step_sizes * descent_direction

class AdagradFull(AdaptiveStepSize):
    """ADAGRAD algorithm for adaptive step-sizes, originally for the more general problem
    of adaptive proximal functions in subgradient methods. This is an implementation of
    the full matrix variation.

    From the paper:
    John Duchi, Elad Hazan, Yoram Singer, 2010
    Adaptive Subgradient Methods for Online Learning and Stochastic Optimization.
    """
    name = "AdagradFull"
    def init_stepsize(self, weights_shape, params):
        self.step_sizes = numpy.ones(weights_shape) * self.alpha
        self.h = numpy.eye(self.step_sizes.size) * params.setdefault("adagrad_precond", 0.001)
        self.adagrad_counter = 0.

    def rescale_update(self, phi_t, phi_tp, delta, reward, descent_direction):
        self.adagrad_counter += 1
        g = descent_direction.flatten()
        self.h = matrix.SMInv(self.h, g, g, 1.)
        if self.adagrad_counter > 0:
            Hinv = numpy.real(scipy.linalg.sqrtm(self.h))
            descent_direction = numpy.dot(Hinv, descent_direction.flatten())
            descent_direction *= numpy.sqrt(self.adagrad_counter)
        return self.step_sizes * descent_direction.reshape(self.step_sizes.shape)


class AdagradDiagonal(AdaptiveStepSize):
    """ADAGRAD algorithm for adaptive step-sizes, originally for the more general problem
    of adaptive proximal functions in subgradient methods. This is an implementation of
    the diagonal matrix variation.

    From the paper:
    John Duchi, Elad Hazan, Yoram Singer, 2010
    Adaptive Subgradient Methods for Online Learning and Stochastic Optimization.
    """
    name = "AdagradDiagonal"
    def init_stepsize(self, weights_shape, params):
        self.step_sizes = numpy.ones(weights_shape) * self.alpha
        self.h = numpy.zeros(weights_shape)
        self.adagrad_counter = 0.

    def rescale_update(self, phi_t, phi_tp, delta, reward, descent_direction):
        self.adagrad_counter += 1
        self.h += descent_direction**2
        if self.adagrad_counter > 1:
            self.step_sizes.fill(self.alpha)
            non_zeros = numpy.where(self.h != 0.0)
            self.step_sizes[non_zeros] *= numpy.sqrt(self.adagrad_counter) /  numpy.sqrt(self.h[non_zeros])
        return self.step_sizes * descent_direction

class AlmeidaAdaptive(AdaptiveStepSize):
    """Adaptive vector step-size.

    From the paper:
    Luis B. Almeida, Thibault Langlois, Jose D. Amaral, and Alexander Plakhov. 1999.
    Parameter adaptation in stochastic optimization
    """
    name = "Almeida"
    def init_stepsize(self, weights_shape, params):
        self.step_sizes = numpy.ones(weights_shape) * self.alpha
        self.prev_grad = None
        self.v = numpy.ones((numpy.prod(weights_shape),))
        self.almeida_gamma = params.setdefault('almeida_gamma', 0.999)
        self.almeida_stepsize = params.setdefault('almeida_stepsize', 0.00001)

    def rescale_update(self, phi_t, phi_tp, delta, reward, descent_direction):
        self.v *= self.almeida_gamma
        self.v += (1. - self.almeida_gamma) * (descent_direction**2).ravel()

        if self.prev_grad is None:
            self.prev_grad = descent_direction.flatten()
        else:
            vbar = self.v.copy()
            vbar[vbar == 0] = 1.0
            self.step_sizes *= (1. + self.almeida_stepsize * numpy.dot(self.prev_grad, descent_direction.ravel()) / vbar).reshape(self.step_sizes.shape)
            self.prev_grad = descent_direction.flatten()

        return self.step_sizes * descent_direction

    @classmethod
    def agent_parameters(cls):
        param_set = super(AlmeidaAdaptive, cls).agent_parameters()
        add_parameter(param_set, "almeida_gamma", default=0.999)
        add_parameter(param_set, "almeida_stepsize", default=0.00001)
        return param_set


class vSGD(AdaptiveStepSize):
    """vSGD is an adaptive step-size algorithm for noisy quadratic objective functions in
    stochastic approximation.

    From the paper:
    Tom Schaul, Sixin Zhang, and Yann LeCun, 2013
    No More Pesky Learning Rates.
    """
    name = "vSGD"
    def init_stepsize(self, weights_shape, params):
        self.step_sizes = numpy.ones(weights_shape) * self.alpha
        self.g = numpy.zeros(weights_shape)
        self.v = numpy.zeros(weights_shape) # For element wise learning rate mode
        self.h = numpy.zeros(weights_shape)
        self.l = 0.0 # For global learning rate mode
        self.t = numpy.ones(weights_shape) * params.setdefault("vsgd_initmeta", 100.)
        self.slow_start = params.setdefault("vsgd_slowstart", 10)
        self.C = params.setdefault("C", 10.)
        self.slowcount = 0

    @classmethod
    def agent_parameters(cls):
        param_set = super(vSGD, cls).agent_parameters()
        add_parameter(param_set, "vsgd_slowstart", default=10, type=int, min=0, max=500)
        add_parameter(param_set, "vsgd_initmeta", default=100., min=1., max=1000.)
        add_parameter(param_set, "vsgd_C", default=10., min=1., max=1000.)
        return param_set

    def rescale_update(self, phi_t, phi_tp, delta, reward, descent_direction):
        # Estimate hessian... somehow..
        est_hessian = (self.gamma * phi_tp - phi_t)**2
        self.update_estimates(est_hessian, descent_direction)
        return self.step_sizes * descent_direction

    def update_stepsize(self):
        # Lets not divide by zero...
        non_zeros = numpy.where(self.v != 0.)
        denom = self.h*self.v*self.C # Overestimate v by a factor of C

        # Step-size adaptation update
        self.step_sizes[non_zeros] = ((self.g[non_zeros]**2) / denom[non_zeros])
        self.t[non_zeros] *= (-(self.g[non_zeros]**2/self.v[non_zeros] - 1.))
        self.t += 1.

    def update_estimates(self, est_hessian, gradient):
        if self.slow_start <= 0:
            self.g *= -(1./self.t - 1.)
            self.g += (1./self.t) * gradient

            self.v *= -(1./self.t - 1.)
            self.v += (1./self.t) * gradient**2

            self.l *= -(1./self.t.max() - 1.)
            self.l += (1./self.t.max()) * numpy.linalg.norm(gradient.ravel())**2

            self.h *= -(1./self.t - 1.)
            self.h += (1./self.t) * est_hessian

            # Bounding condition number, to keep te step-sizes from diverging due to
            # numerical issues...
            self.h = self.h.clip(min=1.)
            self.update_stepsize()
        else:
            # During slow start, compute empirical means and don't change parameters too much
            # Since the notion of 'too much' is like another parameter, we are just
            # going to not move parameters at all until the slow start estimates are done.
            self.step_sizes.fill(0.0)
            self.slowcount += 1
            self.slow_start -= 1
            self.v += gradient**2
            self.g += gradient
            self.l += numpy.linalg.norm(gradient.ravel())**2
            self.h += est_hessian
            if self.slow_start <= 0:
                self.v /= self.slowcount
                self.g /= self.slowcount
                self.h /= self.slowcount
                self.l /= self.slowcount


class vSGDGlobal(vSGD):
    name = "vSGD-g"
    def update_stepsize(self):
        # Lets not divide by zero...
        non_zeros = numpy.where(self.v != 0.)
        denom = self.h*self.v*self.C # Overestimate v by a factor of C

        # Step-size adaptation update
        self.step_sizes.fill(numpy.linalg.norm(self.g.ravel())**2 / (self.h.max() * self.l * self.C))
        self.t *= (-(numpy.linalg.norm(self.g.ravel())**2/self.l - 1.))
        self.t += 1.

class InvMaxEigen(AdaptiveStepSize):
    """The optimal scalar step-size can be proven, under certain assumptions,
    to be one over the maximum eigenvalue of the Hessian of objective function.
    This step-size algorithm is derived by combining the Taylor expansion of the
    objective function, and the power method for extracting eigenvectors.

    From the paper:
    Yann LeCun, Patrice Simard, and Barak Pearlmutter. 1993.
    Automatic learning rate maximization by on-line estimation of the Hessian's eigenvectors.
    """
    name = "InvMaxEigen"
    def init_stepsize(self, weights_shape, params):
        self.step_sizes = numpy.ones(weights_shape) * self.alpha
        self.est_eigvector = numpy.random.normal(size=weights_shape) # Initialize randomly and normalize
        self.est_eigvector /= numpy.linalg.norm(self.est_eigvector.ravel())
        self.lecun_gamma = params.setdefault('lecun_gamma', 0.01) # Convergence rate / accuracy trade off
        self.lecun_alpha = params.setdefault('lecun_alpha', 0.01) # Small -> better estimate, but possible numerical instability
        self.stability_threshold = params.setdefault('lecun_threshold', 0.001)

    @classmethod
    def agent_parameters(cls):
        param_set = super(InvMaxEigen, cls).agent_parameters()
        add_parameter(param_set, "lecun_gamma", default=0.01)
        add_parameter(param_set, "lecun_alpha", default=0.01)
        add_parameter(param_set, "lecun_threshold", default=0.001)
        return param_set

    def rescale_update(self, phi_t, phi_tp, delta, reward, descent_direction):
        # Previous Max Eigen Value Estimate
        prevEigValue = numpy.linalg.norm(self.est_eigvector.ravel())

        # Computing the perterbed gradient estiamte
        perterb_weights = (self.weights + self.lecun_alpha * (self.est_eigvector / prevEigValue))
        perterbed_delta = numpy.dot(perterb_weights.ravel(), (self.gamma * phi_tp - phi_t).ravel()) + reward
        # desc_dir / delta gives just the vector part of the gradient, which doesn't change
        # But, note that descent_direction is already the negative gradient, so
        # we need to switch it back to being the gradient.
        pert_gradient = -perterbed_delta * (phi_t - self.gamma * phi_tp)
        gradient = -descent_direction

         # Update the eigenvector estimate and test for stability
        self.est_eigvector *= (1.0 - self.lecun_gamma)
        self.est_eigvector += (self.lecun_gamma / self.lecun_alpha) * (pert_gradient - gradient)
        update_ratio = numpy.abs(prevEigValue - numpy.linalg.norm(self.est_eigvector.ravel())) / prevEigValue

        if update_ratio <= self.stability_threshold:
            # Update the step-sizes with our newly converged max eigenvalue
            self.alpha = 1./numpy.linalg.norm(self.est_eigvector.ravel())
            self.step_sizes.fill(self.alpha)

            # Reset the estimated eigenvector and star the process again.
            #self.est_eigvector = numpy.random.normal(size=self.est_eigvector.shape)
            #self.est_eigvector /= numpy.linalg.norm(self.est_eigvector.ravel())

        return self.step_sizes * descent_direction

