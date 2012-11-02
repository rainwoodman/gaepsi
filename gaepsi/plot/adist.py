from numpy import sqrt, sin, cos
from numpy import arcsin, arctan2
from numpy import pi, NaN
from numpy import isscalar
from numpy import histogram2d

def adist(field, bins=(40,80), weights=1.0, \
    meridian=0, equator= 0, rrange=[0, None]):
  """
    angular distribution of a field. 
    returns the mean field in an angular region, 
    and the number of samples, and the edges in both direction.
    sinusoid mapping is used. points that there are no samples
    are set to NaN.
  """
  pos = field['locations']
  boxsize = field.boxsize
  center = field.origin + field.boxsize / 2
  x = pos[:, 0] - center[0]
  y = pos[:, 1] - center[1]
  z = pos[:, 2] - center[2]
  
  if isscalar(rrange): 
    rmin = 0
    rmax = rrange
  else :
    rmin = rrange[0]
    rmax = rrange[1]

  if rmax == None : rmax = boxsize.min()/2

  r = sqrt(x**2 + y ** 2 + z**2)
  mask = (rmin < r) & (r < rmax)

  x = x[mask]
  y = y[mask]
  z = z[mask]
  r = r[mask]

  xx,yy = sinusoid(x,y,z, meridian=meridian, equator=equator)

  weights = field[weights][mask]

  print xx.max(), xx.min(), weights,yy.max(), yy.min()
  N, xe, ye = histogram2d(xx, yy, bins=bins, weights=weights)
  print xe, ye
  field = field['default'][mask]

  h, xe, ye  = histogram2d(xx, yy, bins=(xe,ye), weights=weights * field)

  h[N == 0] = NaN

  h /= N

  return h, N, xe, ye

def sinusoid(x,y,z,meridian=0.0, equator=0.0):
  "returns the sinusoical mapping given 3d points in x,y,z"
  r = sqrt(x**2 + y ** 2 + z**2)
  xp = x * cos(equator) + z * sin(equator)
  zp = z * cos(equator) - x * sin(equator)
  phi = arcsin(zp/r)

  theta = arctan2(y, xp)
  theta -= meridian
  theta[theta < - pi] += 2 * pi
  theta[theta > pi] -= 2 * pi

  xx = theta * cos(phi)
  yy = phi
  return xx,yy
