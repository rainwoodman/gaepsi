import numpy
import numpy.random

def sphere2d(size=1):
  """ phi ~[0, 2pi), [ra] theta~ [-pi/2, pi/2),[dec] returns (phi, theta)"""
  phi = numpy.random.uniform(0, 2 * pi, size=size)
  theta = numpy.random.uniform(-1, 1, size=size)
  numpy.acrsin(theta, theta)
  return phi, theta

def discrete(weight, uniform):
  """ sample discrete distribution a uniform random variable """
  cdf = numpy.cumsum(weight, dtype='f8')
  cdf /= cdf[-1]
  uniform = numpy.asarray(uniform)
  return cdf.searchsorted(uniform.ravel(), 'right').reshape(uniform.shape)

def pdf(pdf, range, uniform, bins=100):
  """ sample a pdf from a uniform random variable.
      the pdf is discretized by bins """
  if numpy.isscalar(bins):
    bins = numpy.linspace(range[0], range[1], bins)
  weight = pdf(bins)
  cdf = numpy.zeros_like(weight, dtype='f8')
  cdf[1:] = numpy.cumsum(weight[:-1], dtype='f8')
  cdf /= cdf[-1]
  uniform = numpy.asarray(uniform)
  return numpy.interp(uniform.ravel(), cdf, bins).reshape(uniform.shape)


