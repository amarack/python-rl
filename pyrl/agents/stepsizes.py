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
        self.lmbda = 0.0 # Autostep cannot be used with eligibility traces (in current form)
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
	

