
# Simplest Gaussian RBF implementation ever.

import numpy

class RBFBasis:
    def __init__(self, nvars, num_functions, beta, ranges):
        self.beta = beta
        ranges = numpy.array(ranges)
        self.centers = numpy.random.uniform(ranges[:,0], ranges[:,1].T, (num_functions,nvars))
                                
    def computeFeatures(self, features):
        features = numpy.array(features)
        return numpy.array([numpy.exp(-self.beta * numpy.linalg.norm(features-c)**2) for c in self.centers])


        

