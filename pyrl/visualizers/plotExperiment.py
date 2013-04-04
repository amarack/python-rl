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

def printUsage():
    print "Usage: python -m pyrl.visualizers.plotExperiment title [reward|steps|time] window_size experiment_data1.dat label1 experiment_data2.dat ..."
    sys.exit(1)

def movingaverage(interval, window_size):
    window = numpy.ones(int(window_size))/float(window_size)
    return numpy.convolve(interval, window, 'same')

def processFile(filename, style, windowsize):
    episodes = {}
    maxEp = 0
    numRuns = 0

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
    print "Processed", numRuns, "runs from", filename
    return data

if __name__=="__main__":
    import matplotlib.pyplot as plt
    styles = {"reward": 3, "steps": 1, "time": 2}
    style_labels = {3:"Reward", 1:"Steps", 2:"Time (Seconds)"}

    # Read filename for experiment dump file from arguments
    if len(sys.argv) < 6:
        printUsage()

    mainTitle = sys.argv[1]
    style_str = sys.argv[2].lower()
    style = None
    try:
        style = styles[style_str]
        style_str = style_labels[style]
    except:
        printUsage()

    windowsize = int(sys.argv[3])
    filenames = [sys.argv[i] for i in range(4, len(sys.argv), 2)]
    labels = [sys.argv[i] for i in range(5, len(sys.argv), 2)]
    colors = ['r', 'b', 'g', 'm']
    linestyles = ['-', '--', '-.']
    markers = ["*", ".", "o", ">", 'x', '^', 'H', 'd']
    indx = 0
    for filename in filenames:
        data = processFile(filename, style, windowsize)
        #plt.errorbar(range(data.shape[0]), data[:,0], yerr=data[:,1])
        plt.fill_between(range(data.shape[0]), data[:,0]-data[:,1], data[:,0]+data[:,1], alpha=0.4, color=colors[indx%len(colors)])
        plt.plot(data[:,0], linewidth=2, color=colors[indx%len(colors)], linestyle=linestyles[indx%len(linestyles)], marker=markers[indx%len(markers)], markersize=10)
        indx+=1

    plt.xlabel("Episodes")
    plt.ylabel(style_str)
    plt.title(mainTitle)
    plt.legend(labels,loc='best')
    plt.savefig(mainTitle + ".pdf")
    #plt.show()
        






