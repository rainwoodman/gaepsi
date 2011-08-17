from numpy import array, ones_like, zeros
from numpy import sqrt, int32
from numpy import arange
from gaepsi.kernel import k0
from gaepsi.kernel import kline
from gaepsi.ccode import scanline

def sightline(field, src, dir, dist, npixels=100) :
  """ returns the field density samples along a sightline"""
  tree = field.tree
  i = tree.trace(src, dir, dist)
  density = zeros(dtype='f4', shape=npixels)
  xHI = zeros(dtype='f4', shape=npixels)

  mass = field['mass'][i]
  sml = field['sml'][i]
  locations = field['locations'][i]
  xHI = field['xHI'][i]

  scanline(targets=[density, xHI], values=[mass, xHI * mass], sml = sml, locations =locations, src = src, dir = dir, dist = dist)
  xHI /= mass
  return xHI
