import numpy, itertools

# Compute the value of (A + uv^T)^-1 given A^-1, u, and v. 
# Uses the Sherman-Morrison formula
def SMInv(Ainv, u, v, e):
    u = u.reshape((len(u),1))
    v = v.reshape((len(v),1))
    if e is not None:
        g = numpy.dot(Ainv, u) / (e + numpy.dot(v.T, numpy.dot(Ainv, u)))			
        return (Ainv / e) - numpy.dot(g, numpy.dot(v.T, Ainv/e))
    else:
        return Ainv - numpy.dot(Ainv, numpy.dot(numpy.dot(u,v.T), Ainv)) / ( 1 + numpy.dot(v.T, numpy.dot(Ainv, u)))


def vector_angle(u, v):
    return numpy.arccos(numpy.dot(u,v)/(numpy.linalg.norm(u)*numpy.linalg.norm(v)))*180.0/numpy.pi
