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
import csv, sys, json
from pyrl.misc.json import convert

def movingaverage(interval, window_size):
    interval = numpy.array([interval[0]]*window_size + interval.tolist() + [interval[-1]]*window_size)
    window = numpy.ones(int(window_size))/float(window_size)
    return numpy.convolve(interval, window, 'same')[window_size:-window_size]

def processFile(filename, style, verbose=True, method=None, windowsize=1, kmeans_k=10):
    episodes = {}
    maxEp = 0
    numRuns = 0
    styles = {"reward": 3, "steps": 1, "time": 2}
    style = styles[style]
    diverged = False
    with open(filename, "r") as f:
        csvread = csv.reader(f)
        for line in csvread:
            if int(line[styles["steps"]]) < 0: # Indicates divergence
                diverged = True
            episodes.setdefault(int(line[0]), []).append(float(line[style]))
            maxEp = max(maxEp, int(line[0]))

    numRuns = len(episodes[maxEp])
    data = numpy.zeros((maxEp+1,numRuns))
    for k in episodes.keys():
        current = numpy.array(episodes[k])
        data[k,:] = current[:numRuns]

    if diverged:
        data.fill(data.min())

    locs, means, stdevs = None, None, None
    if method == 'sum':
        # Return just the total return over all episodes
        # This measures average lifetime return (without normalizing)
        data = data.sum(0)
        means = [data.mean()]
        stdevs = [data.std()]
        locs = [0]
    elif method == 'final':
        # Return only the return on the last episode of the experiment
        # This measures the final converged policy
        means = [data[-1,:].mean()]
        stdevs = [data[-1,:].std()]
        locs = [len(data)-1]
    elif method == 'kmeans':
        # Do KMeans, what the hell is going on here?
        # Here's the idea. If we perform a HUGE parameter exploration
        # and hold on to every episode's return, then the amount of data you
        # get is also *huge*. So, what we really want is to downsample the data,
        # without losing the overall picture of what the performance looks like.
        # Instead of actually downsampling, or choosing some fixed schedule of
        # episodes that will be used (and averaging the episodes between them),
        # I went with something a bit smarter.
        #
        # This performs k-means on the data formed by episode indices and episode returns.
        # Thus, we get clusters that have similar returns and are close in episode indices.
        # Then these episode index cluster centers are used as the 'key points', and the
        # rest of the code is just to compute the mean return for each cluster/key-point.
        from sklearn.cluster import KMeans
        means = numpy.array(zip(range(data.shape[0]), data.mean(1)))
        stdevs = data.std(1)
        kmeans_k = min(kmeans_k, maxEp+1)
        estimator = KMeans(init='k-means++', n_clusters=kmeans_k, n_init=10)
        estimator.fit(means)
        key_points = numpy.unique(map(int, estimator.cluster_centers_[:,0])).tolist() # Test/check this line is getting the episode indices
        key_points.sort()
        midpoints = numpy.unique([0] + [(a - b)/2 + b for a,b in zip(key_points[1:], key_points[:-1])] + [len(means)]).tolist()
        results = numpy.array([[means[midpoints[i]:midpoints[i+1],1].mean(), stdevs[midpoints[i]:midpoints[i+1]].mean(), key_points[i]] for i in range(len(midpoints)-1)])
        means = results[:,0]
        stdevs = results[:,1]
        locs = results[:,2]
    else: # None
        # Keeps all the episode data, but allows a sliding window average over it.
        # This is best for smaller experiments, but if the number of trials
        # is in the hundreds of thousands and many episodes each,
        # it generates A LOT of data.
        means = movingaverage(data.mean(1), windowsize)
        stdevs = movingaverage(data.std(1), windowsize)
        locs = numpy.array(range(len(data)))

    if verbose:
        print "Processed", numRuns, "runs from", filename
    return numpy.array(locs), numpy.array(means), numpy.array(stdevs)


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
    parser.add_argument("--json", type=str, action='append',
                        help="Filename of json output from randomized experiments.", default=[])
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
            plt.fill_between(data[:,0], data[:,1]-data[:,2], data[:,1]+data[:,2],
                             alpha=0.4, color=colors[indx%len(colors)])
        plt.plot(data[:,0], data[:,1], linewidth=3, color=colors[indx%len(colors)],
                 linestyle=linestyles[indx%len(linestyles)], marker=markers[indx%len(markers)],
                 markersize=7, markevery=mark_freq)


    for file, label in map(tuple, args.raw):
        labels.append(label)
        locs, means, stdvs = processFile(file, style, windowsize=windowsize)
        drawResult(numpy.array([locs, means, stdvs]).T)
        indx+=1

    for file, label in map(tuple, args.means):
        labels.append(label)
        data = numpy.genfromtxt(file, delimiter=',')
        data[:,1] = movingaverage(data[:,1], windowsize)
        data[:,2] = movingaverage(data[:,2], windowsize)
        drawResult(data)
        indx+=1

    for file in args.json:
        with open(file, 'r') as f:
            for json_line in f:
                result = json.loads(json_line, object_hook=convert)
                data = numpy.array([result['experiment']['episodes'],
                                    result['experiment']['returns'],
                                    result['experiment']['deviations']]).T
                labels.append(result['agent']['name'])
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







