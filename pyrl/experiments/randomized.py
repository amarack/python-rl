
# Author: Will Dabney

import csv, os, json, numpy
from pyrl.misc.timer import Timer
from pyrl.rlglue import RLGlueLocal as RLGlueLocal
from pyrl.rlglue.registry import register_experiment
import rlglue.RLGlue as rl_glue
from pyrl.experiments.episodic import Episodic
import pyrl.visualizers.plotExperiment as plotExperiment

@register_experiment
class RandomizedTrial(Episodic):
    name = "Randomized Trial"

    def __init__(self, **kwargs):
        if not kwargs.has_key('agent') or not kwargs.has_key('environment'):
            print "ERROR: RandomizedTrial must be run locally in order to randomize parameters."
            import sys
            sys.exit(1)

        self.num_trials = kwargs.setdefault('num_trials', 1)
        Episodic.__init__(self, **kwargs)

    def run_experiment(self, filename=None, **args):
        for trial in range(self.num_trials):
            parameters = self.agent.randomize_parameters(**args)
            tmp_file = "rndtrial" + str(numpy.random.randint(1.e10)) + ".dat"
            Episodic.run_experiment(self, filename = tmp_file)

            # Collect results
            score, std = plotExperiment.processFileSum(tmp_file, self.evaluate, 1, verbose=False)
            line = [score, std] + parameters
            if filename is None:
                print ','.join(map(str, line))
            else:
                with open(filename, "a") as f:
                    csvwrite = csv.writer(f)
                    csvwrite.writerow(line)
            os.remove(tmp_file)

