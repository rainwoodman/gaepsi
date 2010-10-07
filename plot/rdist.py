from numpy import sqrt, log10
from numpy import linspace, logspace
from numpy import histogram
from numpy import isscalar

def rdist(field, bins=40, weights=1.0, logscale=False):
  """
    radial distribution of a field. average weighted by weight.
    return the average, the number of samples, and the bin edges
  """
  pos = field['locations']
  boxsize = field.boxsize
  c = boxsize / 2.0

  R = sqrt((pos[:,0] - c)**2 + (pos[:,1] - c)**2 + (pos[:,2] -c )**2)
  if isscalar(bins) :
    if logscale :
      e = logspace(log10(boxsize/500.0), log10(boxsize / 2.0), bins)
    else :
      e = linspace(0, boxsize / 2.0, bins)
  else :
    e = bins

  weights = field[weights]
  field = field['default']

  [N, e] = histogram(R, bins=e, weights = weights)
  [h, e] = histogram(R, bins=e, weights = weights * field)

  h = h / N

  return h, N, e
