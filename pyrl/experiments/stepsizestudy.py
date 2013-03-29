
from pyrl.environments import gridworld,mountaincar,acrobot
from pyrl.agents import sarsa_lambda,stepsizes
import pyrl.experiments.episodic as episodic

import numpy

# spread can be a single int which gives the spread for all variables
# or can be an array with different spreads for each..
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
    if domain == "mountaincar":
        return mountaincar.MountainCar()
    elif domain == "acrobot":
        return acrobot.Acrobot()
    else:
        return gridworld.Gridworld(10,10,10,10)

# Return the class, parameters for sarsa only, filled dictionary for other params
def getStepSize(stepsize, seed=None):
    sarsa_ranges = [[0., 0.1], # epsilon
                    [0., 1.0], # alpha
                    [0.9, 1.0], # gamma
                    [0.4, 1.0]] # lambda

    sarsa_spreads = [0, 1, 0, 0.25]
    sarsa_params = {'basis': 'fourier'}

    if stepsize == "autostep":
        p = obtainParameters(numpy.array(sarsa_ranges), spread=numpy.array(sarsa_spreads), seed=seed)
        return stepsizes.Autostep, p, sarsa_params
    elif stepsize == "alphabound":
        p = obtainParameters(numpy.array(sarsa_ranges), spread=numpy.array(sarsa_spreads), seed=seed)
        p[1] = 1.0 # Force alpha = 1
        return stepsizes.AlphaBounds, p, sarsa_params
    elif stepsize == "rprop":
        sarsa_ranges += [[0., 0.5], # eta_low
                         [0.5, 1.2]] # eta_high
        sarsa_spreads += [2, -2]
        p = obtainParameters(numpy.array(sarsa_ranges), spread=numpy.array(sarsa_spreads), seed=seed)
        sarsa_params['rprop_eta_low'] = p[-2]
        sarsa_params['rprop_eta_high'] = p[-1]
        return stepsizes.RProp, p[:-2], sarsa_params
    elif stepsize == "stc":
        sarsa_ranges += [[1, 1.0e10], # c, convergence rate
                         [1000, 1.0e8]] # N, convergence pivot
        sarsa_spreads += [0, 0]
        p = obtainParameters(numpy.array(sarsa_ranges), spread=numpy.array(sarsa_spreads), seed=seed)
        sarsa_params['stc_c'] = p[-2]
        sarsa_params['stc_N'] = p[-1]
        return stepsizes.STC, p[:-2], sarsa_params
    else: #fixed
        p = obtainParameters(numpy.array(sarsa_ranges), spread=numpy.array(sarsa_spreads), seed=seed)
        class FixedStep:
            pass
        return FixedStep, p, sarsa_params



if __name__=="__main__":
    import argparse
    domains = ["mountaincar", "acrobot", "gridworld"]
    algorithms = ["fixed", "autostep", "rprop", "alphabound"]

    parser = argparse.ArgumentParser(description='Run a step-size algorithm experiment.')
    parser.add_argument("--domain", choices=domains,
                        help="The domain to run the experiment on.")
    parser.add_argument("--algorithm", choices=algorithms, required=True,
                        help="The step-size algorithm to use for this experiment.")
    parser.add_argument("--seed", type=int, default=numpy.random.randint(1.e12), help="Force the seed value.")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to run per trial.")
    parser.add_argument("--num_trials", type=int, default=1, help="Number of trials to run in this experiment.")
    parser.add_argument("--episode_length", type=int, default=5000, help="Maximum number of steps to run per episode.")
    
    args = parser.parse_args()
    agent_class, parameters, named_params = getStepSize(args.algorithm, seed=args.seed)

    print "Algorithm:", args.algorithm
    print "Seed:", args.seed
    print "Parameters:", parameters
    print "Named Params:", named_params
    
    if args.domain is None:
        print "Query only done. Specify a domain to run experiment."
        import sys
        sys.exit()

    env_class = getDomain(args.domain)
    print "Domain:", args.domain

    class AdaptiveSarsa(agent_class, sarsa_lambda.sarsa_lambda):
        def __init__(self, p, params):
            sarsa_lambda.sarsa_lambda.__init__(self, *p, params=params, softmax=False)

    episodic.setupExperiment(args.episode_length,env_class, AdaptiveSarsa(parameters, named_params))
    for run in range(args.num_trials):
        print "Run", run
        episodic.runTrial(args.num_episodes, ("%s_%s_%d.dat") % (args.domain, args.algorithm,args.seed), True)
    print "Done."

