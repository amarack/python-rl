
from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.utils import TaskSpecVRLGLUE3
from pyrl.rlglue.registry import register_agent

from random import Random
import numpy, time
import copy
import sys
import pyrl.basis.fourier as fourier
import pyrl.basis.rbf as rbf
import pyrl.basis.tilecode as tilecode
import pyrl.basis.trivial as trivial
import stepsizes

from skeleton_agent import *


@register_agent
class sarsa_lambda(skeleton_agent):
    name = "Sarsa"

    def init_parameters(self):
        # Initialize algorithm parameters
        self.epsilon = self.params.setdefault('epsilon', 0.1)
        self.alpha = self.params.setdefault('alpha', 0.01)
        self.lmbda = self.params.setdefault('lmbda', 0.7)
        self.gamma = self.params.setdefault('gamma', 1.0)
        self.fa_name = self.params.setdefault('basis', 'trivial')
        self.softmax = self.params.setdefault('softmax', False)
        self.basis = None

    @classmethod
    def agent_parameters(cls):
        param_set = super(sarsa_lambda, cls).agent_parameters()
        add_parameter(param_set, "alpha", default=0.01, help="Step-size parameter")
        add_parameter(param_set, "epsilon", default=0.1, help="Exploration rate for epsilon-greedy, or rescaling factor for soft-max.")
        add_parameter(param_set, "gamma", default=1.0, help="Discount factor")
        add_parameter(param_set, "lmbda", default=0.7, help="Eligibility decay rate")

        # Parameters *NOT* used in parameter optimization
        add_parameter(param_set, "softmax", optimize=False, type=bool, default=False, help="Use soft-max policies")
        add_parameter(param_set, "basis", optimize=False, type=str, help="Basis to use with linear function approximation",
                    choices=['trivial', 'fourier', 'rbf', 'tile'], default='trivial')
        add_parameter(param_set, "fourier_order", optimize=False, default=3, type=int, min=1, max=15)
        add_parameter(param_set, "rbf_number", optimize=False, default=0, type=int, min=0, max=500)
        add_parameter(param_set, "rbf_beta", optimize=False, default=0.9)
        add_parameter(param_set, "tile_number", optimize=False, default=100, type=int, min=0, max=500)
        add_parameter(param_set, "tile_weights", optimize=False, default=2**11, type=int, min=1, max=2**15)
        return param_set

    def agent_supported(self, parsedSpec):
        if parsedSpec.valid:
            # Check observation form, and then set up number of features/states
            assert len(parsedSpec.getDoubleObservations()) + len(parsedSpec.getIntObservations()) > 0, "Expecting at least one continuous or discrete observation"

            # Check action form, and then set number of actions
            assert len(parsedSpec.getIntActions())==1, "Expecting 1-dimensional discrete actions"
            assert len(parsedSpec.getDoubleActions())==0, "Expecting no continuous actions"
            assert not parsedSpec.isSpecial(parsedSpec.getIntActions()[0][0]), "Expecting min action to be a number not a special value"
            assert not parsedSpec.isSpecial(parsedSpec.getIntActions()[0][1]), "Expecting max action to be a number not a special value"
            return True
        else:
            return False

    def agent_init(self,taskSpec):
        """Initialize the RL agent.

        Args:
            taskSpec: The RLGlue task specification string.
        """

        # (Re)initialize parameters (incase they have been changed during a trial
        self.init_parameters()
        # Parse the task specification and set up the weights and such
        TaskSpec = TaskSpecVRLGLUE3.TaskSpecParser(taskSpec)
        if not self.agent_supported(TaskSpec):
            print "Task Spec could not be parsed: "+taskSpecString;
            sys.exit(1)

        self.numStates=len(TaskSpec.getDoubleObservations())
        self.discStates = numpy.array(TaskSpec.getIntObservations())
        self.numDiscStates = int(reduce(lambda a, b: a * (b[1] - b[0] + 1), self.discStates, 1.0))
        self.numActions=TaskSpec.getIntActions()[0][1]+1
        if self.numStates == 0:
            # Only discrete states
            self.numStates = 1
            if self.fa_name != "trivial":
                print "Selected basis requires at least one continuous feature. Using trivial basis."
                self.fa_name = "trivial"

        # Set up the function approximation
        if self.fa_name == 'fourier':
            self.basis = fourier.FourierBasis(self.numStates, TaskSpec.getDoubleObservations(),
                                    order=self.params.setdefault('fourier_order', 3))
        elif self.fa_name == 'rbf':
            num_functions = self.numStates if self.params.setdefault('rbf_number', 0) == 0 else self.params['rbf_number']
            self.basis = rbf.RBFBasis(self.numStates, TaskSpec.getDoubleObservations(),
                                    num_functions=num_functions,
                                    beta=self.params.setdefault('rbf_beta', 0.9))
        elif self.fa_name == 'tile':
            self.basis = tilecode.TileCodingBasis(self.numStates, TaskSpec.getDoubleObservations(),
                                    num_tiles=self.params.setdefault('tile_number', 100),
                                    num_weights=self.params.setdefault('tile_weights', 2048))
        else:
            self.basis = trivial.TrivialBasis(self.numStates, TaskSpec.getDoubleObservations())

        self.weights = numpy.zeros((self.numDiscStates, self.basis.getNumBasisFunctions(), self.numActions))
        self.traces = numpy.zeros(self.weights.shape)
        self.init_stepsize(self.weights.shape, self.params)

        self.lastAction=Action()
        self.lastObservation=Observation()


    def getAction(self, state, discState):
        """Get the action under the current policy for the given state.

        Args:
            state: The array of continuous state features
            discState: The integer representing the current discrete state value

        Returns:
            The current policy action, or a random action with some probability.
        """

        if self.softmax:
            return self.sample_softmax(state, discState)
        else:
            return self.egreedy(state, discState)

    def sample_softmax(self, state, discState):
        Q = None
        Q = numpy.dot(self.weights[discState,:,:].T, self.basis.computeFeatures(state))
        Q -= Q.max()
        Q = numpy.exp(numpy.clip(Q/self.epsilon, -500, 500))
        Q /= Q.sum()

        Q = Q.cumsum()
        return numpy.where(Q >= numpy.random.random())[0][0]

    def egreedy(self, state, discState):
        if self.randGenerator.random() < self.epsilon:
            return self.randGenerator.randint(0,self.numActions-1)
        return numpy.dot(self.weights[discState,:,:].T, self.basis.computeFeatures(state)).argmax()

    def getDiscState(self, state):
        """Return the integer value representing the current discrete state.

        Args:
            state: The array of integer state features

        Returns:
            The integer value representing the current discrete state
        """

        if self.numDiscStates > 1:
            x = numpy.zeros((self.numDiscStates,))
            mxs = self.discStates[:,1] - self.discStates[:,0] + 1
            mxs = numpy.array(list(mxs[:0:-1].cumprod()[::-1]) + [1])
            x = numpy.array(state) - self.discStates[:,0]
            #print (x*mxs).sum()
            return (x * mxs).sum()
        else:
            return 0

    def agent_start(self,observation):
        """Start an episode for the RL agent.

        Args:
            observation: The first observation of the episode. Should be an RLGlue Observation object.

        Returns:
            The first action the RL agent chooses to take, represented as an RLGlue Action object.
        """

        theState = numpy.array(list(observation.doubleArray))
        thisIntAction=self.getAction(theState, self.getDiscState(observation.intArray))
        returnAction=Action()
        returnAction.intArray=[thisIntAction]

        # Clear traces
        self.traces.fill(0.0)

        self.lastAction=copy.deepcopy(returnAction)
        self.lastObservation=copy.deepcopy(observation)
        return returnAction

    def update_traces(self, phi_t, phi_tp):
        self.traces *= self.gamma * self.lmbda
        self.traces += phi_t

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

        # Update eligibility traces
        phi_t = numpy.zeros(self.traces.shape)
        phi_tp = numpy.zeros(self.traces.shape)
        phi_t[lastDiscState, :, lastAction] = self.basis.computeFeatures(lastState)
        phi_tp[newDiscState, :, newIntAction] = self.basis.computeFeatures(newState)

        self.update_traces(phi_t, phi_tp)
        self.update(phi_t, phi_tp, reward)

        returnAction=Action()
        returnAction.intArray=[newIntAction]

        self.lastAction=copy.deepcopy(returnAction)
        self.lastObservation=copy.deepcopy(observation)
        return returnAction


    def init_stepsize(self, weights_shape, params):
        self.step_sizes = numpy.ones(weights_shape) * self.alpha

    def rescale_update(self, phi_t, phi_tp, delta, reward, descent_direction):
        return self.step_sizes * descent_direction

    def update(self, phi_t, phi_tp, reward):
        # Compute Delta (TD-error)
        delta = numpy.dot(self.weights.flatten(), (self.gamma * phi_tp - phi_t).flatten()) + reward

        # Update the weights with both a scalar and vector stepsize used
        # Adaptive step-size if that is enabled
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
        phi_tp = numpy.zeros(self.traces.shape)
        phi_t[lastDiscState, :, lastAction] = self.basis.computeFeatures(lastState)

        self.update_traces(phi_t, phi_tp)
        self.update(phi_t, phi_tp, reward)

    def agent_cleanup(self):
        """Perform any clean up operations before the end of an experiment."""
        pass

    def has_diverged(self):
        value = self.weights.sum()
        return numpy.isnan(value) or numpy.isinf(value)


@register_agent
class residual_gradient(sarsa_lambda):
    """Residual Gradient(lambda) algorithm. This RL algorithm is essentially what Sarsa(labmda)
    would be if you were actually doing gradient descent on the squared Bellman error.

    From the paper (original):
    Residual Algorithms: Reinforcement Learning with Function Approximation.
    Leemon Baird. 1995.
    """

    name = "Residual Gradient"
    def update_traces(self, phi_t, phi_tp):
        self.traces *= self.gamma * self.lmbda
        self.traces += (phi_t - self.gamma * phi_tp)

@register_agent
class fixed_policy(sarsa_lambda):
    """This agent takes a seed from which it generates the weights for the
    state-action value function. It then behaves just like Sarsa but with a
    learning rate of zero (0). Thus, it has a fixed state-action value function
    and thus a fixed policy (which has been randomly generated).
    """

    name = "Fixed Policy"

    def init_parameters(self):
        sarsa_lambda.init_parameters(self)
        self.policy_seed = self.params.setdefault('seed', int(time.time()*10000))

    @classmethod
    def agent_parameters(cls):
        param_set = super(fixed_policy, cls).agent_parameters()
        add_parameter(param_set, "seed", type=int, default=int(time.time()*10000), min=1, max=int(1.4e13))
        return param_set

    def agent_init(self,taskSpec):
        sarsa_lambda.agent_init(self, taskSpec)
        numpy.random.seed(self.policy_seed)
        self.weights = 2.*(numpy.random.random(self.weights.shape) - 0.5)
        numpy.random.seed(None)

    def update(self, phi_t, phi_tp, reward):
        pass

ABSarsa = stepsizes.genAdaptiveAgent(stepsizes.AlphaBounds, sarsa_lambda)

if __name__=="__main__":
    runAgent(sarsa_lambda)


