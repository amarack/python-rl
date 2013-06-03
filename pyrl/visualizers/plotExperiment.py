################################################################################
# plotExperiment.py
# Author: Will Dabney
#
# A script to plot the output of results from one or more PyRL experiments.
# It allows plotting reward, steps, and time in seconds per episode using a
# sliding window average (or with a window size of 1, which results in no
# averaging). Each experiment should be contained in a single data file,
# and within each file one or more trials/runs of that experiment can be
# contained. If more than one trial is contained in a single data file
# the plotted results will be an average over all trials for that experiment
# with standard deviations shown by a shaded region around the mean.
#
# Example: python -m pyrl.visualizers.plotExperiment "A complete test of plotting" reward 5 test.dat Test test2.dat Test2
################################################################################

import numpy
import csv, sys

def movingaverage(interval, window_size):
    interval = numpy.array([interval[0]]*window_size + interval.tolist() + [interval[-1]]*window_size)
    window = numpy.ones(int(window_size))/float(window_size)
    return numpy.convolve(interval, window, 'same')[window_size:-window_size]

def processFile(filename, style, windowsize, verbose=True):
    episodes = {}
    maxEp = 0
    numRuns = 0
    styles = {"reward": 3, "steps": 1, "time": 2}
    style = styles[style]

    with open(filename, "r") as f:
        csvread = csv.reader(f)
        for line in csvread:
            episodes.setdefault(int(line[0]), []).append(float(line[style]))
            maxEp = max(maxEp, int(line[0]))

    data = numpy.zeros((maxEp+1,2))
    numRuns = len(episodes[0])
    for k in episodes.keys():
        current = numpy.array(episodes[k])
        data[k,0] = current.mean()
        data[k,1] = current.std()

    data[:,0] = movingaverage(data[:,0], windowsize)
    data[:,1] = movingaverage(data[:,1], windowsize)
    if verbose:
        print "Processed", numRuns, "runs from", filename
    return data


def processFileSum(filename, style, windowsize, verbose=True):
    episodes = {}
    maxEp = 0
    numRuns = 0
    styles = {"reward": 3, "steps": 1, "time": 2}
    style = styles[style]

    with open(filename, "r") as f:
        csvread = csv.reader(f)
        for line in csvread:
            episodes.setdefault(int(line[0]), []).append(float(line[style]))
            maxEp = max(maxEp, int(line[0]))

    numRuns = len(episodes[0])
    data = numpy.zeros((maxEp+1,numRuns))
    numRuns = len(episodes[0])
    for k in episodes.keys():
        current = numpy.array(episodes[k])
        data[k,:] = current[:numRuns]
    data = data.sum(0)
    if verbose:
        print "Processed", numRuns, "runs from", filename
    return data.mean(), data.std()


if __name__=="__main__":
    import matplotlib.pyplot as plt
    import argparse

    # Labels used for the y-axis depending on the evaluation metric being plotted
    style_labels = {"reward":"Reward", "steps":"Steps", "time":"Time (Seconds)"}

    parser = argparse.ArgumentParser(description='Plot the results of a randomized ' + \
                                         'parameter search experiment. Use --raw to specify data output from an experiment, ' + \
                                         'or --means if the data has already been processed into means and standard deviations.')
    parser.add_argument("--raw", type=str, action='append', nargs=2,
                        help="Filename of raw collected data from experiments.", default=[])
    parser.add_argument("--means", type=str, action='append', nargs=2,
                        help="Filename of episode number, means and standard deviations " + \
                            "for each episode of an experiment. ", default=[])
    parser.add_argument("--output", type=str, help="Filename to save the resulting figure.")
    parser.add_argument("--title", type=str, help="Title for the figure.", default="")
    parser.add_argument("--ylabel", type=str, help="Alternative label of y axis for the figure.")
    parser.add_argument("--windowsize", type=int, help="Window size for smoothing.", default=1)
    parser.add_argument("--target", type=str, help="Evaluation target.",
                        default="reward", choices=style_labels.keys())
    parser.add_argument("--legend_loc", type=str, help="Legend location",
                        default="best", choices=["best", "lower right", "lower left",
                                                 "upper right", "upper left"])
    parser.add_argument("--nobars", action='store_true', default=False)
    parser.add_argument("--eplimit", type=int)
    parser.add_argument("--markevery", type=int)
    args = parser.parse_args()

    mainTitle = args.title
    style = args.target

    try:
        style_str = style_labels[style]
    except:
        printUsage()

    if args.ylabel is not None:
        style_str = args.ylabel

    windowsize = args.windowsize
    labels = []
    colors = ['r', 'b', 'g', 'm']
    linestyles = ['-', '--', '-.']
    markers = ["x", ".", "o", ">", '*', '^', 'H', 'd']
    indx = 0

    def drawResult(data):
        mark_freq = windowsize
        if args.eplimit is not None:
            data = data[:args.eplimit,:]
        if args.markevery is not None:
            mark_freq = args.markevery

        if not args.nobars:
            plt.fill_between(range(data.shape[0]), data[:,0]-data[:,1], data[:,0]+data[:,1],
                             alpha=0.4, color=colors[indx%len(colors)])
        plt.plot(data[:,0], linewidth=3, color=colors[indx%len(colors)],
                 linestyle=linestyles[indx%len(linestyles)], marker=markers[indx%len(markers)],
                 markersize=7, markevery=mark_freq)


    for file, label in map(tuple, args.raw):
        labels.append(label)
        data = processFile(file, style, windowsize)
        drawResult(data)
        indx+=1

    for file, label in map(tuple, args.means):
        labels.append(label)
        data = numpy.genfromtxt(file)[:,(1,2)]
        data[:,0] = movingaverage(data[:,0], windowsize)
        data[:,1] = movingaverage(data[:,1], windowsize)
        drawResult(data)
        indx+=1

    plt.xlabel("Episodes")
    plt.ylabel(style_str)
    plt.title(mainTitle)
    plt.legend(labels, loc=args.legend_loc)
    if args.output is not None:
        plt.savefig(args.output)
    else:
        plt.show()







