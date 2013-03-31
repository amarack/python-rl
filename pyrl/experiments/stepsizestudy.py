################################################################################
# stepsizestudy.py
# Author: Will Dabney
#
# This script is an experiment for studying the effects of different step-size 
# algorithms when used with Sarsa(lambda). Parameters and meta-parameters 
# are chosen randomly from distributions customized to the appropriate 
# range is distribution of each parameter.
# 
# Results are saved out in the filename: domain_algorithm_seed.dat
# This means we can run additional trials for any previously generated seed/trial. 
#
# Example: python -m pyrl.experiments.stepsizestudy --domain mountaincar --algorithm fixed
################################################################################

from pyrl.environments import gridworld,mountaincar,acrobot
from pyrl.agents import sarsa_lambda,stepsizes
import pyrl.experiments.episodic as episodic
import pyrl.visualizers.plotExperiment as plotExperiment

import numpy
import glob
import csv

def obtainParameters(ranges, spread=0, seed=None):
    """Generates random parameters in the specified ranges. 

    Args:
        ranges: An array of arrays which give the lower and upper limit for the parameters
        [spread=0]: If a float then gives the spread over values for all parameters. If 
                an array of shape (len(ranges),), then gives custom spread for each parameter 
                and allows negative values which are interpeted as biasing towards the high end 
                rather than the low end of the range when the spread gets larger. 
        [seed=None]: Takes an int seed which is used to seed the random generator. Reseeds the 
                random generator randomly after the parameters have been generated. 

    Returns:
        An array of randomly generated parameters.
    """

    numpy.random.seed(seed)
    raw = numpy.random.random((len(ranges),2))
    try:
        len(spread)
        flipped = numpy.where(spread < 0)
        new_raw = raw[:,0] * 10**(raw[:,1] * -spread)
        new_raw[flipped] = 1.0 - (raw[flipped,0] * 10**(raw[flipped,1] * spread[flipped]))
        raw = new_raw
    except TypeError:
        raw = raw[:,0] * 10**(raw[:,1] * -spread)
    numpy.random.seed()
    return raw*(ranges[:,1] - ranges[:,0]) + ranges[:,0]

def getDomain(domain):
    """Get the domain instance given the string name.

    Args:
        domain: String name for domain to get

    Returns:
        An environment instantiated with the domain specified.
    """
    
    if domain == "mountaincar":
        return mountaincar.MountainCar()
    elif domain == "acrobot":
        return acrobot.Acrobot()
    else:
        return gridworld.Gridworld(10,10,10,10)

def getStepSize(stepsize, seed=None, justParams=False):
    """Get the step-size class and parameters given an algorithm name and an optional seed.

    Args:
        stepsize: Step-size algorithm name
        (seed=None): Optional seed to use for random generation
        (justParams=False): Optional, if true will only return the random parameters generated.

    Returns:
        Tuple containing: step-size algorithm class, sarsa parameters, and sarsa named parameters.
    """
    
    # Sarsa parameter ranges
    sarsa_ranges = [[0., 0.1], # epsilon
                    [0., 1.0], # alpha
                    [0.9, 1.0], # gamma
                    [0.4, 1.0]] # lambda

    # Spreads for sarsa parameters
    sarsa_spreads = [0, 1, 0, 0.25]

    # Sarsa named parameters
    sarsa_params = {'basis': 'fourier'}

    if stepsize == "autostep":
        # Autostep, uses initial step-size, but the two other meta
        # parameters can safely be left constant.
        p = obtainParameters(numpy.array(sarsa_ranges), spread=numpy.array(sarsa_spreads), seed=seed)
        if justParams:
            return p
        else:
            return stepsizes.Autostep, p, sarsa_params
    elif stepsize == "alphabound":
        # AlphaBound, forces step-size to start at 1.0 (as this is a nontunable parameter).
        p = obtainParameters(numpy.array(sarsa_ranges), spread=numpy.array(sarsa_spreads), seed=seed)
        p[1] = 1.0 # Force alpha = 1
        if justParams:
            return p
        else:
            return stepsizes.AlphaBounds, p, sarsa_params
    elif stepsize == "rprop":
        # RProp, requires an initial step-size, and low/high step-sizes
        sarsa_ranges += [[0., 0.5], # eta_low
                         [0.5, 1.2]] # eta_high
        sarsa_spreads += [2, -2]
        p = obtainParameters(numpy.array(sarsa_ranges), spread=numpy.array(sarsa_spreads), seed=seed)
        if justParams:
            return p
        else:
            sarsa_params['rprop_eta_low'] = p[-2]
            sarsa_params['rprop_eta_high'] = p[-1]
            return stepsizes.RProp, p[:-2], sarsa_params
    elif stepsize == "stc":
        # Search-Then-Converge, requires two parameters
        sarsa_ranges += [[1, 1.0e10], # c, convergence rate
                         [1000, 1.0e8]] # N, convergence pivot
        sarsa_spreads += [0, 0]
        p = obtainParameters(numpy.array(sarsa_ranges), spread=numpy.array(sarsa_spreads), seed=seed)
        if justParams:
            return p
        else:
            sarsa_params['stc_c'] = p[-2]
            sarsa_params['stc_N'] = p[-1]
            return stepsizes.STC, p[:-2], sarsa_params
    else: 
        # Fixed step-size, requires initial step-size for the constant value
        p = obtainParameters(numpy.array(sarsa_ranges), spread=numpy.array(sarsa_spreads), seed=seed)
        if justParams:
            return p
        else:
            class FixedStep:
                pass
            return FixedStep, p, sarsa_params

def collectData(args):
    """Collect all the matching randomized trials into a matrix of scores with parameters.
    """

    files = glob.glob(("./%s_%s_*.dat") % (args.domain, args.algorithm))
    params = [[plotExperiment.processFile(f, 3, 1)[:,0].sum()] + getStepSize(args.algorithm, seed=int(f.split("_")[-1][:-4]), justParams=True).tolist() for f in files]
    with open(args.collect, "wb") as f:
        csvwr = csv.writer(f)
        csvwr.writerows(params)
    

def runExperiment(args):
    """Run a randomized trial given the parameters from the command line.
    """

    agent_class, parameters, named_params = getStepSize(args.algorithm, seed=args.seed)
    print "Algorithm:", args.algorithm
    print "Seed:", args.seed
    print "Parameters:", parameters
    print "Named Params:", named_params

    # No domain specified, so we just print out the parameters generated
    if args.domain is None:
        print "Query only done. Specify a domain to run experiment."
        import sys
        sys.exit()

    env_class = getDomain(args.domain)
    print "Domain:", args.domain

    # Build the class which combines the step-size with sarsa
    class AdaptiveSarsa(agent_class, sarsa_lambda.sarsa_lambda):
        def __init__(self, p, params):
            sarsa_lambda.sarsa_lambda.__init__(self, *p, params=params, softmax=False)

    # Setup and run the randomized experiment
    episodic.setupExperiment(args.episode_length,env_class, AdaptiveSarsa(parameters, named_params))
    for run in range(args.num_trials):
        print "Run", run
        episodic.runTrial(args.num_episodes, ("%s_%s_%d.dat") % (args.domain, args.algorithm,args.seed), True)
    print "Done."


if __name__=="__main__":
    import argparse
    domains = ["mountaincar", "acrobot", "gridworld"]
    algorithms = ["fixed", "autostep", "rprop", "alphabound", "stc"]

    parser = argparse.ArgumentParser(description='Run a step-size algorithm experiment.')
    parser.add_argument("--domain", choices=domains,
                        help="The domain to run the experiment on.")
    parser.add_argument("--algorithm", choices=algorithms, required=True,
                        help="The step-size algorithm to use for this experiment.")
    parser.add_argument("--seed", type=int, default=numpy.random.randint(1.e12), help="Force the seed value.")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to run per trial.")
    parser.add_argument("--num_trials", type=int, default=1, help="Number of trials to run in this experiment.")
    parser.add_argument("--episode_length", type=int, default=5000, help="Maximum number of steps to run per episode.")
    parser.add_argument("--collect", type=str, help="Collect all the randomized runs in the current directory and produce a matrix of parameters and their scores into the filename given.")

    args = parser.parse_args()
    if args.collect is None:
        runExperiment(args)
    else:
        collectData(args)
