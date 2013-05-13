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
from sklearn import (manifold, decomposition,
                     random_projection)


def plotOneParam(data, title, label, filename=None):
    """Plot performance against values of one parameter.

    Args:
        data: Two dimensional numpy array. First column gives performance values.
        title: Title for the resulting plot
        label: Label for the parameter axis
        (filename=None): Filename to save the figure out as, if None (default) will show the figure.
    """

    data = data[numpy.lexsort((data[:,0],data[:,1]))]

    plt.plot(data[:,1], data[:,0], linewidth=2)
    plt.xlabel(label)
    plt.title(title)
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)

def plotTwoParams(data, title, filename=None, labels=["X", "Y"]):
    """Plot performance against values of two parameters, or a two-dim embedding.

    Args:
        data: Two dimensional numpy array. First column gives performance values.
        title: Title for the resulting plot
        (filename=None): Filename to save the figure out as, if None (default) will show the figure.
        (labels=[X,Y]): List of labels for the two axes.
    """

    stride = 2
    x = numpy.linspace(data[:,1].min(), data[:,1].max())
    y = numpy.linspace(data[:,2].min(), data[:,2].max())
    X,Y = numpy.meshgrid(x, y)
    Z = griddata((data[:,1], data[:,2]), data[:,0], (X, Y),method='linear')
    fig = plt.figure()
    ax = fig.gca()
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])

    surf = ax.contourf(X, Y, Z, zdir='z',cmap=plt.cm.jet)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_title(title)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)


def embededParams(data, title, n_neighbors=2, method='pca', filename=None, labels=["First Principal Component", "Second Principal Component"]):
    """Plot performance against values of two-dimensional embedding of all the parameters.

    Args:
        data: Two dimensional numpy array. First column gives performance values.
        title: Title for the resulting plot
        (n_neighbors=2): Number of neighbors to use for embeddings requiring it.
        (method=pca): Dimensionality reduction method to use. Defaults to PCA.
        (filename=None): Filename to save the figure out as, if None (default) will show the figure.
        (labels=[...]): List of labels for the two axes.
    """

    if method == 'pca':
        X_pca = decomposition.RandomizedPCA(n_components=2).fit_transform(data[:,1:])
    elif method == 'isomap':
        X_pca = manifold.Isomap(n_neighbors=2, n_components=2).fit_transform(data[:,1:])
    elif method == 'lle':
        X_pca = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
                                                method='standard').fit_transform(data[:,1:])
    elif method == 'mds':
        X_pca = manifold.MDS(n_components=2, n_init=1, max_iter=100).fit_transform(data[:,1:])
    else:
        print "Error unknown method"
        return
    plotTwoParams(numpy.array([data[:,0].tolist()] + X_pca.T.tolist()).T,
                  title, filename=filename, labels=labels)

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Plot the results of a randomized ' + \
                                         'parameter search experiment.')
    parser.add_argument("--file", type=str, required=True,
                        help="Filename of collected data from experiments.")
    parser.add_argument("--parameter", type=str, action='append', nargs=2,
                        help="Name of parameters to plot. If greater than two, names are not " + \
                            "used and instead the first two principal components are used. " + \
                            "Give first the index followed by the label of the parameter. " + \
                            "Can be used at most twice.")
    parser.add_argument("--output", type=str, help="Filename to save the resulting figure.")
    parser.add_argument("--method", type=str, default='pca', choices=['pca', 'isomap', 'lle', 'mds'],
                        help="Choose a method for dimensionality reduction. Only used when more " + \
                            "than two parameters, and no parameter flag is given.")
    parser.add_argument("--title", type=str, help="Title for the figure.",
                        default="Parameter Exploration")
    args = parser.parse_args()

    # Get the collected data
    data = numpy.genfromtxt(args.file, delimiter=',')

    # Determine if we need to do an embedding or just plot the data
    if args.parameter is None or len(args.parameter) > 2:
        if data.shape[1] > 2:
            embededParams(data, args.title, method=args.method, filename=args.output)
        else:
            plotOneParam(data, args.title, "Parameter", filename=args.output)
    else:
        if len(args.parameter) == 1:
            plotOneParam(data[:,(0,int(args.parameter[0][0]))], args.title,
                         args.parameter[0][1], filename=args.output)
        else: #length is 2
            plotTwoParams(data[:,tuple([0] + map(lambda k: int(k[0]), args.parameter))],
                          args.title, filename=args.output,
                          labels=map(lambda k: k[1], args.parameter))

