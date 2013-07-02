
# Author: Will Dabney

import csv, os, json, numpy, copy
from pyrl.misc.timer import Timer
from pyrl.rlglue import RLGlueLocal as RLGlueLocal
from pyrl.rlglue.registry import register_experiment
import rlglue.RLGlue as rl_glue
from pyrl.experiments.episodic import Episodic
import pyrl.visualizers.plotExperiment as plotExperiment
from pyrl.misc.parameter import *


@register_experiment
class RandomizedTrial(Episodic):
    name = "Randomized Trial"

    def __init__(self, config, **kwargs):
        if not kwargs.has_key('agent') or not kwargs.has_key('environment'):
            print "ERROR: RandomizedTrial must be run locally in order to randomize parameters."
            import sys
            sys.exit(1)

        self.num_trials = kwargs.setdefault('num_trials', 1)
        self.evaluate = kwargs.setdefault('evaluate', 'reward') #reward, steps, time
        self.eval_reduce = kwargs.setdefault('evaluate_reduce', 'sum') # None, 'sum', 'final', 'kmeans'
        self.k = kwargs.setdefault('kmeans_k', 10)
        Episodic.__init__(self, config, **kwargs)

    def run_experiment(self, filename=None):
        param_parser = self.agent.agent_parameters()
        for trial in range(self.num_trials):
            parameters = copy.deepcopy(self.configuration['agent']['params'])
            # Randomize the parameters, those marked not optimizable get their default
            for name, value in randomize_parameters(param_parser):
                # Then, set the parameter value, but only if not already set
                parameters.setdefault(name, value)

            # Set params for current agent
            self.agent.params = parameters

            # Run a trial...
            tmp_file = "rndtrial" + str(numpy.random.randint(1.e10)) + ".dat"
            Episodic.run_experiment(self, filename = tmp_file)

            # Collect results
            locs, means, std = plotExperiment.processFile(tmp_file, self.evaluate, verbose=False, method=self.eval_reduce, kmeans_k=self.k)
            json_out = copy.deepcopy(self.configuration)
            json_out['agent']['params'] = parameters
            json_out['experiment']['episodes'] = locs.tolist()
            json_out['experiment']['returns'] = means.tolist()
            json_out['experiment']['deviations'] = std.tolist()

            if filename is None:
                print json.dumps(json_out)
            else:
                with open(filename, "a") as f:
                    f.write(json.dumps(json_out) + "\n")
            os.remove(tmp_file)

