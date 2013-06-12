
# Author: Will Dabney

from random import Random
import numpy

import pyrl.basis.fourier as fourier
import pyrl.basis.rbf as rbf
import pyrl.basis.tilecode as tilecode
import pyrl.basis.trivial as trivial
from planner import Planner

from sklearn import linear_model
from sklearn.svm import SVR
from sklearn import tree

class FittedQIteration(Planner):
    """FittedQIteration is an implementation of the Fitted Q-Iteration algorithm of Ernst, Geurts, Wehenkel (2005).

    This class allows the use of a variety of regression algorithms, provided by scikits-learn, to be used for
    representing the Q-value function. Additionally, different basis functions can be applied to the features before
    being passed to the regressors, including trivial, fourier, tile coding, and radial basis functions.
    """

    def __init__(self, model, **kwargs):
        """Inits the Fitted Q-Iteration planner with discount factor, instantiated model learner, and additional parameters.

        Args:
            model: The model learner object
            gamma=1.0: The discount factor for the domain
            **kwargs: Additional parameters for use in the class.
        """
        Planner.__init__(self, model, **kwargs)
        self.fa_name = self.params.setdefault('basis', 'trivial')
        self.params.setdefault('iterations', 200)
        self.params.setdefault('support_size', 200)
        self.basis = None

        # Set up regressor
        learn_name = self.params.setdefault('regressor', 'linreg')
        if learn_name == 'linreg':
            self.learner = linear_model.LinearRegression()
        elif learn_name == 'ridge':
            self.learner = linear_model.Ridge(alpha = self.params.setdefault('l2', 0.5))
        elif learn_name == 'tree':
            self.learner = tree.DecisionTreeRegressor()
        elif learn_name == 'svm':
            self.learner = SVR()
        else:
            self.learner = None

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
        self.randParameter('iterations', args, sample=numpy.random.randint(500))
        self.randParameter('support_size', args, sample=numpy.random.randint(500))
        # Randomize basis parameters
        if self.fa_name == 'fourier':
            self.randParameter('fourier_order', args, sample=numpy.random.randint(1,5)*2 + 1)
        elif self.fa_name == 'rbf':
            self.randParameter('rbf_number', args, sample=numpy.random.randint(100))
            self.randParameter('rbf_beta', args)
        elif self.fa_name == 'tile':
            self.randParameter('tile_number', args, sample=numpy.random.randint(200))
            self.randParameter('tile_weights', args, sample=2**numpy.random.randint(15))
        return super(FittedQIteration,self).randomize_parameters(**args)

    def planner_init(self, numDiscStates, contFeatureRanges, numActions, rewardRange):
        self.has_plan = False
        self.ranges, self.actions = self.model.getStateSpace()
        # Set up basis
        if self.fa_name == 'fourier':
            self.basis = fourier.FourierBasis(len(self.ranges), self.ranges,
                            order=self.params.setdefault('fourier_order', 3))
        elif self.fa_name == 'rbf':
            self.basis = rbf.RBFBasis(len(self.ranges), self.ranges,
                            num_functions=self.params.setdefault('rbf_number', len(self.ranges)),
                            beta=self.params.setdefault('rbf_beta', 1.0))
        elif self.fa_name == 'tile':
            self.basis = tilecode.TileCodingBasis(len(self.ranges), self.ranges,
                            num_tiles=self.params.setdefault('tile_number', 100),
                            num_weights=self.params.setdefault('tile_weights', 2048))
        else:
            self.basis = trivial.TrivialBasis(len(self.ranges), self.ranges)


    def getStateAction(self, state, action):
        """Returns the basified state feature array for the given state action pair.

        Args:
            state: The array of state features
            action: The action taken from the given state

        Returns:
            The array containing the result of applying the basis functions to the state-action.
        """
        state = self.basis.computeFeatures(state)
        stateaction = numpy.zeros((self.actions, len(state)))
        stateaction[action,:] = state
        return stateaction.flatten()

    def predict(self, state, action):
        """Predict the next state, reward, and termination probability for the current state-action.

        Args:
            state: The array of state features
            action: The action taken from the given state

        Returns:
            Tuple (next_state, reward, termination), where next_state gives the predicted next state,
            reward gives the predicted reward for transitioning to that state, and termination
            gives the expected probabillity of terminating the episode upon transitioning.

            All three are None if no model has been learned for the given action.
        """
        if self.model.has_fit[action]:
            return self.model.predict(state, action)
        else:
            return None, None, None

    def getValue(self, state):
        """Get the Q-value function value for the greedy action choice at the given state (ie V(state)).

        Args:
            state: The array of state features

        Returns:
            The double value for the value function at the given state
        """
        if self.has_plan:
            return self.learner.predict([self.getStateAction(state, a) for a in range(self.actions)]).max()
        else:
            return None

    def getAction(self, state):
        """Get the action under the current plan policy for the given state.

        Args:
            state: The array of state features

        Returns:
            The current greedy action under the planned policy for the given state. If no plan has been formed,
            return a random action.
        """
        if self.has_plan:
            return self.learner.predict([self.getStateAction(state, a) for a in range(self.actions)]).argmax()
        else:
            return self.randGenerator.randint(0, self.actions-1)


    def updatePlan(self):
        """Run Fitted Q-Iteration on samples from the model, and update the plan accordingly."""
        for sample_iter in range(self.params.setdefault('resample', 1)):
            self.has_plan = False
            prev_coef = None
            samples = self.model.sampleStateActions(self.params['support_size'])
            outcomes = self.model.predictSet(samples)

            Xp = []
            X = []
            R = []
            gammas = []
            for a in range(self.actions):
                Xp += map(lambda k: [self.getStateAction(k, b) for b in range(self.actions)], outcomes[a][0])
                X += map(lambda k: self.getStateAction(k, a), samples[a])
                R += list(outcomes[a][1])
                gammas += list((1.0 - outcomes[a][2]) * self.gamma)

            Xp = numpy.array(Xp)
            Xp = Xp.reshape(Xp.shape[0]*Xp.shape[1], Xp.shape[2])
            X = numpy.array(X)
            R = numpy.array(R)
            gammas = numpy.array(gammas)
            targets = []
            Qp = None

            error = 1.0
            iter2 = 0
            threshold = 1.0e-4
            while error > threshold and iter2 < self.params['iterations']:
                if self.has_plan:
                    Qprimes = self.learner.predict(Xp).reshape((X.shape[0], self.actions))
                    targets = R + gammas*Qprimes.max(1)
                    Qp = Qprimes
                else:
                    targets = R
                    self.has_plan = True

                self.learner.fit(X, targets)

                try:
                    if prev_coef is not None:
                        error = numpy.linalg.norm(prev_coef - self.learner.coef_)
                    prev_coef = self.learner.coef_.copy()
                except:
                    pass

                iter2 += 1

            #print "#?", sample_iter, iter2, error, self.model.exp_index
            if error <= threshold:
                return
