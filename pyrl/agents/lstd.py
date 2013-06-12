# Author: Will Dabney

from random import Random
import numpy

from rlglue.agent import AgentLoader as AgentLoader
from pyrl.rlglue.registry import register_agent

import pyrl.misc.matrix as matrix
import sarsa_lambda, qlearning

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

    def randomize_parameters(self, **args):
        """Generate parameters randomly, constrained by given named parameters.

        If used, this must be called before agent_init in order to have desired effect.

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
        # LSTD does not use alpha, so we remove it from the list
        param_list = sarsa_lambda.sarsa_lambda.randomize_parameters(**args)
        self.update_freq = self.params.setdefault('lstd_update_freq', numpy.random.randint(200))
        return param_list[:1] + param_list[2:] + [self.update_freq]

    def init_parameters(self):
        sarsa_lambda.sarsa_lambda.init_parameters(self)
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
        sarsa_lambda.sarsa_lambda.init_parameters(self)
        self.lstd_gamma = self.gamma
        self.gamma = 1.0


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
        LSTD.init_parameters(self)
        self.num_sweeps = int(self.params.setdefault('ilstd_sweeps', 1))

    def randomize_parameters(self, **args):
        """Generate parameters randomly, constrained by given named parameters.

        If used, this must be called before agent_init in order to have desired effect.

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
        param_list = sarsa_lambda.sarsa_lambda.randomize_parameters(**args)
        self.num_sweeps = int(args.setdefault('ilstd_sweeps', numpy.random.randint(99)+1))
        return param_list + [self.num_sweeps]

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
        sarsa_lambda.sarsa_lambda.init_parameters(self)
        self.delta = self.params.setdefault('rlstd_delta', 1.0)

    def randomize_parameters(self, **args):
        """Generate parameters randomly, constrained by given named parameters.

        If used, this must be called before agent_init in order to have desired effect.

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
        param_list = sarsa_lambda.sarsa_lambda.randomize_parameters(**args)
        self.delta = args.setdefault('rlstd_delta', numpy.random.randint(1000)+1)
        return param_list + [self.delta]

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

    def randomize_parameters(self, **args):
        """Generate parameters randomly, constrained by given named parameters.

        If used, this must be called before agent_init in order to have desired effect.

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
        # LSTD does not use alpha, so we remove it from the list
        param_list = super(LSTDQ, self).randomize_parameters(**args)
        self.update_freq = self.params.setdefault('lstd_update_freq', numpy.random.randint(200))
        self.num_samples = self.params.setdefault('lstd_num_samples', numpy.random.randint(5000))
        self.precond = self.params.setdefault('lstd_precond', numpy.random.random())
        return param_list[:1] + param_list[2:] + [self.update_freq, self.num_samples, self.precond]

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
    def randomize_parameters(self, **args):
        param_list = super(LSPI, self).randomize_parameters(**args)
        self.threshold = self.params.setdefault('lspi_threshold', numpy.random.random())
        return param_list + [self.threshold]


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
    import argparse
    parser = argparse.ArgumentParser(description='Run Least Squares Temporal Difference Learning agent.')
    parser.add_argument("--epsilon", type=float, default=0.1, help="Probability of exploration with epsilon-greedy.")
    parser.add_argument("--softmax", type=float,
                help="Use softmax policies with the argument giving tau, the divisor which scales values used when computing soft-max policies.")
    parser.add_argument("--stepsize", "--alpha", type=float, default=0.01,
                help="The step-size parameter which affects how far in the direction of the gradient parameters are updated. Only with iLSTD.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor")
    parser.add_argument("--mu", type=float, default=1., help="Forgetting factor for RLS-TD")
    parser.add_argument("--beta", type=float, default=0.01, help="Online LSTD initializes A^-1 to I + beta*RandMatrix.")
    parser.add_argument("--lambda", type=float, default=0.7, help="The eligibility traces decay rate. Set to 0 to disable eligibility traces.", dest='lmbda')
    parser.add_argument("--num_sweeps", type=int, default=1, help="Number of sweeps to perform per step in iLSTD.")
    parser.add_argument("--delta", type=float, default=200.,
                help="Value to initialize diagonal matrix to, for inverse matrix, in RLS-TD.")
    parser.add_argument("--algorithm", choices=["lstd", "online", "ilstd", "rlstd"],
                default="lstd", help="Set the LSTD algorithm to use. LSTD, Online LSTD, iLSTD, or Recursive LSTD.")
    parser.add_argument("--basis", choices=["trivial", "fourier", "tile", "rbf"], default="trivial",
                help="Set the basis to use for linear function approximation.")
    parser.add_argument("--fourier_order", type=int, default=3, help="Order for Fourier basis")
    parser.add_argument("--rbf_num", type=int, default=10, help="Number of radial basis functions to use.")
    parser.add_argument("--rbf_beta", type=float, default=1.0, help="Beta parameter for radial basis functions.")
    parser.add_argument("--tiles_num", type=int, default=100, help="Number of tilings to use with Tile Coding.")
    parser.add_argument("--tiles_size", type=int, default=2048, help="Memory size, number of weights, to use with Tile Coding.")

    args = parser.parse_args()
    params = {}
    params['alpha'] = args.alpha
    params['gamma'] = args.gamma
    params['lmbda'] = args.lmbda

    if args.softmax is not None:
        params['softmax'] = True
        params['epsilon'] = args.softmax
    else:
        params['softmax'] = False
        params['epsilon'] = args.epsilon

    params['basis'] = args.basis
    params['fourier_order'] = args.fourier_order
    params['rbf_number'] = args.rbf_num
    params['rbf_beta'] = args.rbf_beta
    params['tile_number'] = args.tiles_num
    params['tile_weights'] = args.tiles_size

    if args.algorithm == 'rlstd':
        params['alpha'] = args.mu
        params['rlstd_delta'] = args.delta
        AgentLoader.loadAgent(RLSTD(**params))
    elif args.algorithm == 'online':
        params['alpha'] = args.beta
        AgentLoader.loadAgent(oLSTD(**params))
    elif args.algorithm == 'ilstd':
        params['ilstd_sweeps'] = args.num_sweeps
        AgentLoader.loadAgent(iLSTD(**params))
    else:
        AgentLoader.loadAgent(LSTD(**params))
