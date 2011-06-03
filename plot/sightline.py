from numpy import array, ones_like, zeros
from numpy import sqrt, int32
from numpy import arange
from gaepsi.kernel import k0
from gaepsi.kernel import kline

def sightline(field, x0,y0, npixels=100) :
  """ returns the field density samples along a sightline"""
  boxsize = field.boxsize

  quadtree = field.quadtree()

  plist = quadtree.list(x0, y0)
  pixelsize = boxsize[2] / npixels

  pos = field['locations']
  x = pos[:,0][plist]
  y = pos[:,1][plist]
  z = pos[:,2][plist]
  sml = field['sml'][plist]
  values = field['default'][plist]

  dx = x - x0
  mask = dx > boxsize[0] /2
  dx[mask] = boxsize[0] - dx[mask]
  dy = y - y0
  mask = dy > boxsize[1] /2
  dy[mask] = boxsize[1] - dy[mask]
  dists = sqrt(dx ** 2 + dy ** 2)

  mask = dists < sml
  x = x[mask]
  y = y[mask]
  z = z[mask]
  sml = sml[mask]
  values = values[mask]
  dists = dists[mask]


  deltas = sqrt(sml**2 - dists**2)

  line = zeros(npixels)

  sums = kline(dists/sml) / sml ** 2

  print 'particles =', len(x)
  for ip in arange(len(x)):
    pz = arange(z[ip] - deltas[ip] - pixelsize /2.0, z[ip] + deltas[ip] + pixelsize /2.0, pixelsize)
    pzp = int32(pz / pixelsize)
    mask = ((pzp < npixels) & (pzp >=0))

    eta = sqrt(dists[ip]**2 + (pz - z[ip])**2) / sml[ip]
    density = k0(eta) / sml[ip] ** 3
    sum = density.sum() * pixelsize
    if sum > 0.0:
      fac = sums[ip] / sum
      density *= fac

    add = values[ip] * density
    line[pzp[mask]] += add[mask]
  return arange(npixels) * pixelsize, line

