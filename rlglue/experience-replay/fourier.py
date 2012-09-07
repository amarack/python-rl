import numpy, itertools
class FourierBasis:
    def __init__(self, nvars, order, ranges):
        nterms = pow(order + 1.0, nvars)
        self.numTerms = nterms
        self.order = order
        self.ranges = numpy.array(ranges)
        iter = itertools.product(''.join(map(str, range(order+1))), repeat=nvars)
        self.multipliers = numpy.array([list(map(int,x)) for x in iter])

    def scale(self, value, pos):
        return (value - self.ranges[pos,0]) / (self.ranges[pos,1] - self.ranges[pos,0])

    def computeFeatures(self, features):
        basisFeatures = numpy.array([self.scale(features[i],i) for i in range(len(features))])
        return numpy.cos(numpy.pi * numpy.dot(self.multipliers, basisFeatures))
        
