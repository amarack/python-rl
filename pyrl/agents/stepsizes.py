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


def genAdaptiveAgent(stepsize_class, agent_class):
    """Generate an RL agent by combining an existing agent with a step-size algorithm."""

    @register_agent
    class AdaptiveAgent(stepsize_class, agent_class):
        name = "Adaptive (" + stepsize_class.name + ") " + agent_class.name
	def __init__(self, **args):
            agent_class.__init__(self, **args)

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
		param_list = agent_class.randomize_parameters(self, **args)
                param_list += stepsize_class.randomize_parameters(self, **args)
		return param_list

    return AdaptiveAgent

class AdaptiveStepSize(object):
    name = "Fixed StepSize"

    def init_stepsize(self, weights_shape, params):
        self.step_sizes = numpy.ones(weights_shape) * self.alpha

    def rescale_update(self, phi_t, phi_tp, delta, reward, descent_direction):
        return self.step_sizes * descent_direction

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
        return []



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

    def randomize_parameters(self, **args):
        self.params['ghs_a'] = args.setdefault('ghs_a', numpy.random.random()*10000.)
        return [self.params['ghs_a']]

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

    def randomize_parameters(self, **args):
        self.params['mcclain_a'] = args.setdefault('mcclain_a', numpy.random.random()*self.alpha)
        return [self.params['mcclain_a']]

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

    def randomize_parameters(self, **args):
        self.params['stc_c'] = args.setdefault('stc_c', numpy.random.random()*1.e10)
        self.params['stc_N'] = args.setdefault('stc_N', numpy.random.random()*1.e6)
        return [self.params['stc_c'], self.params['stc_N']]


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

    def randomize_parameters(self, **args):
        self.params['rprop_eta_high'] = args.setdefault('rprop_eta_high', numpy.random.random()*2.)
        self.params['rprop_eta_low'] = args.setdefault('rprop_eta_low', numpy.random.random()*self.params['rprop_eta_high'])
        return [self.params['rprop_eta_low'], self.params['rprop_eta_high']]


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
        x = phi_t.flatten()
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

    def randomize_parameters(self, **args):
        self.params['autostep_mu'] = args.setdefault('autostep_mu', numpy.random.random())
        self.params['autostep_tau'] = args.setdefault('autostep_tau', numpy.random.random()*1.e6)
        return [self.params['autostep_mu'], self.params['autostep_tau']]

class AlphaBounds(AdaptiveStepSize):
    """AlphaBounds adaptive scalar step-size.

    From the paper:
    Dabney, W. and A. G. Barto (2012).
    Adaptive Step-Size for Online Temporal Difference Learning.
    """
    name = "AlphaBound"
    def init_stepsize(self, weights_shape, params):
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
        self.v = numpy.zeros(weights_shape)
        self.h = numpy.zeros(weights_shape)
        self.t = numpy.ones(weights_shape) * params.setdefault("vsgd_initmeta", 100.)
        self.slow_start = params.setdefault("vsgd_slowstart", 50)

    def randomize_parameters(self, **args):
        self.params['vsgd_slowstart'] = args.setdefault('vsgd_slowstart', numpy.random.randint(500))
        self.params['vsgd_initmeta'] = args.setdefault('vsgd_initmeta', float(numpy.random.randint(1000)))
        return [self.params['vsgd_slowstart'], self.params['vsgd_initmeta']]

    def rescale_update(self, phi_t, phi_tp, delta, reward, descent_direction):
        # Estimate hessian... somehow..
        est_hessian = (descent_direction/delta)**2
        self.g *= -(1./self.t - 1.)
        self.g += (1./self.t) * descent_direction

        self.v *= -(1./self.t - 1.)
        self.v += (1./self.t) * descent_direction**2

        self.h *= -(1./self.t - 1.)
        self.h += (1./self.t) * est_hessian
        denom = self.h*self.v
        denom[denom==0] = 1.0
        self.step_sizes = (self.g**2) / denom

        if self.slow_start <= 0:
            self.t *= (-((self.g**2 / self.v) - 1.)).clip(1.e-15, 1.)
            self.t += 1.

        self.slow_start -= 1
        return self.step_sizes * descent_direction



