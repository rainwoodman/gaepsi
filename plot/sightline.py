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

  pos = f['locations'][pars]
  sml = f['sml'][pars]
  mass = f['mass'][pars]
  massHI = f['xHI'][pars] * mass * 0.76
  vzHI = f['vel'][pars, 2] * massHI
  tempHI = f.cosmology.ie2T(0.76, f['ie'][pars], f['ye'][pars]) * massHI

  LrhoHI = zeros(shape=npixels, dtype='f8')

  LvzHI = zeros_like(LrhoHI)
  LtempHI = zeros_like(LrhoHI)

  scanline(locations = pos, sml = sml, targets = [LrhoHI, LvzHI, LtempHI], values = [massHI, vzHI, tempHI], src = src, dir = dir, L = f.boxsize[2])

  Lvred = linspace(0, f.boxsize[2], LrhoHI.size) * f.cosmology.H(a=1.0)

  LnHI = LrhoHI / f.cosmology.units.PROTONMASS
  LtauHI = LnHI * f.cosmology.units.LYMAN_ALPHA_CROSSSECTION / f.cosmology.H(a=1.0) * f.cosmology.units.C

  LvzHI /= LrhoHI
  LtempHI /= LrhoHI
  LvthermHI = sqrt(LtempHI * f.cosmology.units.BOLTZMANN / f.cosmology.units.PROTONMASS)

  M = (3.1416 * 2) ** -0.5 / LvthermHI[newaxis, :] * exp(- 0.5 * (Lvred[:, newaxis] - (LvzHI + Lvred)[newaxis, :]) ** 2 / (LvthermHI ** 2)[newaxis, :]) * diff(Lvred)[0]

#M /= sum(M, axis = 1)[newaxis, :]

  LtauHIred= inner(M, LtauHI)
  return LtauHIred, LtauHI
