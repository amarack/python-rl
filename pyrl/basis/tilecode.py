
import numpy
from CTiles import tiles

class TileCodingBasis:
    def __init__(self, num_tiles, num_weights):
        self.num_tiles = num_tiles
        self.mem_size = num_weights

    def computeFeatures(self, features):
        indices = tiles.tiles(self.num_tiles, self.mem_size, list(features))
        result = numpy.zeros((self.mem_size,))
        result[indices] = 1.0
        return result



