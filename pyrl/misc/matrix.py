import numpy, itertools, math

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

# Modified version of this solution:
# http://stackoverflow.com/questions/11615664/multivariate-normal-density-in-python
# Takes the inverse of the covariance matrix instead of the covariance matrix
def mvnpdf(x, mu, sigma_inv):
    size = len(x)
    if size == len(mu) and sigma_inv.shape == (size, size):
        det = 1./numpy.linalg.det(sigma_inv)
        norm_const = 1.0/ ( math.pow((2*numpy.pi),float(size)/2) * math.pow(det,0.5) )
        x_mu = x - mu
        result = math.pow(math.e, -0.5 * numpy.dot(x_mu, numpy.dot(sigma_inv, x_mu)))
        return norm_const * result
    else:
        raise NameError("The dimensions of the input don't match")

