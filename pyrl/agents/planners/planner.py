
from random import Random
import numpy

class Planner(object):

    def __init__(self, model, **kwargs):
        self.model = model
        self.gamma = kwargs.setdefault('gamma', 1.0)
        self.params = kwargs
        self.randGenerator = Random()


    def planner_init(self, numDiscStates, contFeatureRanges, numActions, rewardRange):
        pass

    def randParameter(self, param_key, args, sample=None):
        """A utility function for use inside randomize_parameters. Takes a parameter
        key (name), the named arguments passed to randomize_parameters, and optionally
        the sampled random value to set in case the key does not exist in the arguments.

        This will then set it (if not already present) in args and assign which ever value
        args ends up with into params.
        """
        if sample is None:
            sample = numpy.random.random()
        self.params[param_key] = args.setdefault(param_key, sample)

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
        self.randParameter('gamma', args)
        return args

    def updateExperience(self, lastState, action, newState, reward):
        if self.model.updateExperience(lastState, action, newState, reward):
            self.updatePlan()

    def updatePlan(self):
        pass

    def getAction(self, state):
        pass



