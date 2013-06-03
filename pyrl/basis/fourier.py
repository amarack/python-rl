import numpy, itertools

class FourierBasis(object):
    """Fourier Basis linear function approximation. Requires the ranges for each dimension, and is thus able to
    use only sine or cosine (and uses cosine). So, this has half the coefficients that a full Fourier approximation
    would use.

    From the paper:
    G.D. Konidaris, S. Osentoski and P.S. Thomas.
    Value Function Approximation in Reinforcement Learning using the Fourier Basis.
    In Proceedings of the Twenty-Fifth Conference on Artificial Intelligence, pages 380-385, August 2011.
    """

    def __init__(self, nvars, order, ranges):
        nterms = pow(order + 1.0, nvars)
        self.numTerms = nterms
        self.order = order
        self.ranges = numpy.array(ranges)
        iter = itertools.product(range(order+1), repeat=nvars)
        self.multipliers = numpy.array([list(map(int,x)) for x in iter])

    def scale(self, value, pos):
        if self.ranges[pos,0] == self.ranges[pos,1]:
            return 0.0
        else:
            return (value - self.ranges[pos,0]) / (self.ranges[pos,1] - self.ranges[pos,0])

    def computeFeatures(self, features):
        basisFeatures = numpy.array([self.scale(features[i],i) for i in range(len(features))])
        return numpy.cos(numpy.pi * numpy.dot(self.multipliers, basisFeatures))


