from matplotlib.mlab import find
from numpy import array, ones_like, zeros
from numpy import sqrt, int32
from numpy import arange
from gadget.kernel import kernel
from gadget.kernel import kernel_line

def sightline(field, x0,y0, npixels=100) :
  """ returns the field density samples along a sightline"""
  boxsize = field.boxsize

  quadtree = field.quadtree()

  plist = quadtree.list(x0, y0)[0]
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

  d = sqrt(dx ** 2 + dy ** 2)
  delta = sqrt(sml**2 - d**2)

  line = zeros(npixels)
  bad = 0
  for ip in find(d < sml):
    pz = arange(z[ip] - delta[ip]+ pixelsize /2.0, z[ip] + delta[ip] - pixelsize /2.0, pixelsize)
    pzp = int32(pz / pixelsize)
    pzp[pzp >= npixels] = pzp[pzp >= npixels] - npixels
    pzp[pzp < 0] = pzp[pzp < 0] + npixels
    eta = sqrt((d[ip]**2 + (pz - z[ip])**2)) / sml[ip]
    kernels = array([kernel(ETA) for ETA in eta]) / sml[ip] ** 3
    linekernel = kernel_line(d[ip]/sml[ip]) / sml[ip] ** 2
    sumkernel = sum(kernels) * pixelsize
    if  sumkernel > 0.0 :
      fac = linekernel / sumkernel
      if fac > 2 or fac < 0.5: 
        print fac, linekernel, kernels
        bad = bad +1
      kernels *= fac
    else :
      kernels = ones_like(kernels) / kernels.size

    add = values[ip] * kernels
    line[pzp] += add
  print bad, sum(d < sml)
  return arange(npixels) * pixelsize, line

