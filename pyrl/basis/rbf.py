
# Simplest Gaussian RBF implementation ever.
import numpy, trivial

class RBFBasis(trivial.TrivialBasis):
    """Radial Basis Functions basis. This implementation is just about as simplistic as it gets.
    This really could use some work to make it more competitive with state of the art.
    """

    def __init__(self, nvars, ranges, num_functions=10, beta=0.9):
        trivial.TrivialBasis.__init__(self, nvars, ranges)
        self.beta = beta
        self.num_functions = num_functions
        self.centers = numpy.random.uniform(self.ranges[:,0], self.ranges[:,1].T, (self.num_functions,self.numTerms))

    def getNumBasisFunctions(self):
        return self.num_functions

    def computeFeatures(self, features):
        if len(features) == 0:
            return numpy.ones((1,))
        features = numpy.array(features)
        return numpy.array([numpy.exp(-self.beta * numpy.linalg.norm(features-c)**2) for c in self.centers])




