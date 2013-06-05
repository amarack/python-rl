
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
        self.evaluate = kwargs.setdefault('evaluate', 'reward') #reward, steps, time
        self.eval_reduce = kwargs.setdefault('evaluate_reduce', 'sum') # None, 'sum', 'final', 'kmeans'
        self.k = kwargs.setdefault('kmeans_k', 10)
        Episodic.__init__(self, **kwargs)

    def run_experiment(self, filename=None, **args):
        for trial in range(self.num_trials):
            parameters = self.agent.randomize_parameters(**args)
            tmp_file = "rndtrial" + str(numpy.random.randint(1.e10)) + ".dat"
            Episodic.run_experiment(self, filename = tmp_file)

            # Collect results
            locs, means, std = plotExperiment.processFile(tmp_file, self.evaluate, verbose=False, method=self.eval_reduce, kmeans_k=self.k)

            # Line = NumDataPoints, [index, mean, std for each in numdatapoints], parameters
            line = [len(locs)]
            for i in range(len(locs)):
                line += [locs[i], means[i], std[i]]
            line += parameters
            if filename is None:
                print ','.join(map(str, line))
            else:
                with open(filename, "a") as f:
                    csvwrite = csv.writer(f)
                    csvwrite.writerow(line)
            os.remove(tmp_file)

