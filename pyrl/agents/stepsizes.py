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

import numpy


class GHS(object):
    """Generalized Harmonic Stepsize algorithm for scalar step-sizes.

    Follows the equation: a_t = a_0 * (a / a + t - 1), 
    for parameters a_0 and a
    """
    
    def init_stepsize(self, weights_shape, params):
        self.step_sizes = numpy.ones(weights_shape) * self.alpha
        self.last_update = numpy.zeros(weights_shape)
        self.ghs_param = params.setdefault('ghs_a', 10.0)
        self.ghs_counter = 1

    def compute_stepsize(self, phi_t, phi_tp, delta, reward):
        self.step_sizes.fill(self.alpha * self.ghs_param/(self.ghs_param + self.ghs_counter - 1))
        self.ghs_counter += 1

class McClains(object):
    """McClain's formula for scalar step-size

    Follows the equation: a_t = a_{t-1} / (1 + a_{t-1} - a)
    unless t = 0, then use a_0, for parameters a_0 and a
    """
    
    def init_stepsize(self, weights_shape, params):
        self.step_sizes = numpy.ones(weights_shape) * self.alpha
        self.mcclain_param = params.setdefault('mcclain_a', 0.01)

    def compute_stepsize(self, phi_t, phi_tp, delta, reward):
        self.step_sizes.fill(self.alpha)
        self.alpha /= (1 + self.alpha - self.mcclain_param)

class STC(object):
    """Search-Then-Converge formula for scalar step-size

    Follows the equation: a_t = a_{t-1} * (1 + (c/a_0) * (t/N)) / (1 + (c/a_0) * (t/N) + N * (t^2/N^2))
    for parameters a_0 the initial stepsize, c the target stepsize, N the pivot point.
    N (the pivot point) is simply approximately how many steps at which the formula begins to 
    converge more rather than search more.
    """
    
    def init_stepsize(self, weights_shape, params):
        self.step_sizes = numpy.ones(weights_shape) * self.alpha
        self.stc_a0 = self.alpha
        self.stc_c = params.setdefault('stc_c', 1000000.0)
        self.stc_N = params.setdefault('stc_N', 500000.0)
        self.stc_counter = 0

    def compute_stepsize(self, phi_t, phi_tp, delta, reward):
        self.alpha *= (1 + (self.stc_c * self.stc_counter)/(self.stc_a0 * self.stc_N)) 
        self.alpha /= (1 + (self.stc_c * self.stc_counter)/(self.stc_a0 * self.stc_N) + self.stc_N*(self.stc_counter**2)/self.stc_N**2)
        self.step_sizes.fill(self.alpha)
        self.stc_counter += 1

class RProp(object):
    """RProp algorithm for vector step-sizes.

    From the paper:
    Riedmiller, M. and Braun, H. (1993). 
    A direct adaptive method for faster backpropagation learning: The RPROP algorithm.
    """
    
    def init_stepsize(self, weights_shape, params):
        self.step_sizes = numpy.ones(weights_shape) * self.alpha
        self.last_update = numpy.zeros(weights_shape)
        self.eta_low = params.setdefault('rprop_eta_low', 0.01)
        self.eta_high = params.setdefault('rprop_eta_high', 1.2)

    def compute_stepsize(self, phi_t, phi_tp, delta, reward):
        sign_changes = numpy.where(self.last_update * delta * self.traces <= 0)
        self.step_sizes.fill(self.eta_high)
        self.step_sizes[sign_changes] = self.eta_low


class Autostep(object):
    """Autostep algorithm for vector step-sizes.

    From the paper:
    Mahmood, A. R., Sutton, R. S., Degris, T., and Pilarski, P. M. 2012.
    Tuning-free step-size adaptation.
    """
    
    def init_stepsize(self, weights_shape, params):
        self.step_sizes = numpy.ones(weights_shape) * self.alpha
        self.h = numpy.zeros((numpy.prod(weights_shape),))
        self.v = numpy.zeros((numpy.prod(weights_shape),))
        #self.lmbda = 0.0 # Autostep cannot be used with eligibility traces (in current form)
        self.mu = params.setdefault('autostep_mu', 1.0e-2)
        self.tau = params.setdefault('autostep_tau', 1.0e4)

    def compute_stepsize(self, phi_t, phi_tp, delta, reward):
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
        #plus_note[plus_note < 0] = 0.0 # This may or may not be used depending on which paper you read
        self.h = self.h * plus_note + self.step_sizes.flatten()*delta*x

	
class AlphaBounds(object):
    """AlphaBounds adaptive scalar step-size.

    From the paper:
    Dabney, W. and A. G. Barto (2012).
    Adaptive Step-Size for Online Temporal Difference Learning.
    """

    def init_stepsize(self, weights_shape, params):
        self.step_sizes = numpy.ones(weights_shape) * self.alpha
        
    def compute_stepsize(self, phi_t, phi_tp, delta, reward):
        deltaPhi = (self.gamma * phi_tp - phi_t).flatten()
        denomTerm = numpy.dot(self.traces.flatten(), deltaPhi.flatten())
        self.alpha = numpy.min([self.alpha, 1.0/numpy.abs(denomTerm)])
        self.step_sizes.fill(self.alpha)
	


