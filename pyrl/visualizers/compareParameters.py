################################################################################
# plotParameters.py
# Author: Will Dabney
#
# A script to plot the output of the collected results of a randomized
# parameter search experiment.
#
# Using the --parameter index name, argument you can plot a single parameter or
# pair of parameters against their collected performance values. Specifying no
# parameter will result in a low dimensionality embedding of all the parameters
# to be plotted against their values.
# Thus, with one parameter specified we get the usual results graphs used to
# illustrate how an algorithm performs as a parameter varries. With two parameters
# specified we get this same conceptual view but for the interaction of the two
# parameters. And finally with three or more, we get something more interesting
# which shows the overal behavior pattern with respect to parameter change that
# the algorithm exibits (at least on the given domain).
#
# Example: python -m pyrl.visualizers.plotParameters --file exp.dat --parameter 1 alpha
################################################################################

import numpy
import matplotlib.pyplot as plt
import sys
import argparse
from scipy.interpolate import griddata

def loadParameterData(filename, param_index):
    data = numpy.genfromtxt(filename, delimiter=',')[:,(0,1,param_index)] # Grab the mean, std, and parameter
    data = data[numpy.lexsort((data[:,2],)),:]

    if data[:,2].std() <= 1.e-10:
        xs = numpy.linspace(0, 1.0)
        ys = xs.copy()
        ys.fill(data[:,0].mean())
        stdv = xs.copy()
        xs.fill(0.0)
        return numpy.array([ys, stdv, xs]).T
    else:
        return data


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Plot a comparison of algorithms parameter ' + \
                                         'exploration for a singe parameter.')
    parser.add_argument("--file", type=str, action='append', nargs=3, required=True,
                        help="Parameter exploration algorithm name, results file, and " + \
                            "the index of the parameter to display. Ex: Alg algfile.dat 2")
    parser.add_argument("--title", type=str, help="Title for the figure.",
                        default="Parameter Exploration")
    parser.add_argument("--xlabel", type=str, help="Name of parameter being compared, label for x-axis",
                        default="Parameter")
    parser.add_argument("--ylabel", type=str, help="Name of evaluation metric for algorithms. " + \
                            "This is the label for the y-axis", default="Total Return")
    parser.add_argument("--output", type=str, help="Filename to save the resulting figure.")
    parser.add_argument("--nobars", action='store_true', default=False, help="Disable plotting of error bars for standard deviations.")
    args = parser.parse_args()

    for (name, file, index) in args.file:
        data = loadParameterData(file, index)
        if not args.nobars:
            plt.errorbar(data[:,2], data[:,0], yerr=data[:,1])
        else:
            plt.plot(data[:,2], data[:,0])
        plt.hold(True)

    plt.legend(map(lambda k: k[0], args.file), loc='best')
    plt.title(args.title)
    plt.xlabel(args.xlabel)
    plt.ylabel(args.ylabel)
    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()

