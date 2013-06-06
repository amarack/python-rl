
import numpy
import trivial
from CTiles import tiles

class TileCodingBasis(trivial.TrivialBasis):
    """ Tile Coding Basis. From Rich Sutton's implementation,
        http://incompleteideas.net/rlai.cs.ualberta.ca/RLAI/RLtoolkit/tiles.html
    """

    def __init__(self, nvars, ranges, num_tiles=100, num_weights=2048):
        trivial.TrivialBasis.__init__(self, nvars, ranges)
        self.num_tiles = num_tiles
        self.mem_size = num_weights

    def getNumBasisFunctions(self):
        return self.mem_size

    def computeFeatures(self, features):
        if len(features) == 0:
            return numpy.ones((1,))
        features = list(trivial.TrivialBasis.computeFeatures(self, features))
        indices = tiles.tiles(self.num_tiles, self.mem_size, features)
        result = numpy.zeros((self.mem_size,))
        result[indices] = 1.0
        return result



