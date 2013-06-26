
# Author: Will Dabney

import sys
pyrl_path = "###"
sys.path.append(pyrl_path)

import os, numpy
from pyrl.misc.timer import Timer
from pyrl.rlglue import RLGlueLocal as RLGlueLocal
from pyrl.rlglue.registry import register_experiment
import rlglue.RLGlue as rl_glue
from pyrl.experiments.episodic import Episodic
import pyrl.visualizers.plotExperiment as plotExperiment
from pyrl.misc.parameter import *
from pyrl.rlglue.run import fromjson

def main(job_id, params):
    print 'Anything printed here will end up in the output directory for job #:', str(job_id)
    parameters = {}
    for key in params:
        if isinstance(key, unicode):
            parameters[key.encode('utf-8')] = params[key]
    else:
            parameters[key] = params[key]

    for key in parameters:
        parameters[key] = map(lambda k: k.encode('utf-8') if isinstance(k, unicode) else k, parameters[key])
        if len(parameters[key]) == 1:
            value = parameters[key][0]
            try:
                value = float(value)
            except:
                if value.lower() == "false":
                    value = False
                elif value.lower() == "true":
                    value = True

            parameters[key] = value
    print parameters

    my_path = os.path.dirname(os.path.abspath(__file__))
    tmp_file = os.path.join(my_path, "rndtrial" + str(numpy.random.randint(1.e10)) + ".dat")
    my_path = os.path.abspath(os.path.join(my_path, "experiment.json"))
    agent, a_args, env, env_args, exp, exp_args = fromjson(my_path)

    for key in parameters:
        a_args.setdefault(key, parameters[key])

    config = {'agent': {'name': agent.name, 'params': a_args},
              'environment': {'name': env.name, 'params': env_args},
              'experiment': {'name': exp.name, 'params': exp_args}}

    experiment = Episodic(config, agent=agent(**a_args),
                    environment=env(**env_args), **exp_args)

    # Using this try/except makes debugging in spearmint 1mil times easier
    try:
        experiment.run_experiment(filename=tmp_file)
    except Exception as ex:
        import traceback
        traceback.print_exc()

    locs, means, std = plotExperiment.processFile(tmp_file, "reward", verbose=False, method="sum")
    os.remove(tmp_file)
    print "Result:", -means[0]
    return -means[0]


