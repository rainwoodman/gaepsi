"""
Detecting bubble edge(Rs) mostly for stromgren sphere growth
"""

import healpy as hy
from numpy import zeros, zeros_like, arange, linspace, digitize, newaxis, bincount
from numpy import arange, argmax, argmin, where
from numpy import exp

def first(cond, last=None, axis=-1):
  ind = arange(cond.shape[axis])
  notfound = cond.shape[axis]
  if last is not None:
    return argmin(where(cond & (ind[newaxis, :] > last[:, newaxis]), ind, notfound), axis=axis)
  return argmin(where(cond, ind, notfound), axis=axis)

def last(cond, first=None, axis=-1):
  ind = arange(cond.shape[axis])
  notfound = -1
  if first is not None:
    return argmax(where(cond & (ind[newaxis, :] < first[:, newaxis]), ind, notfound), axis=axis)
  return argmax(where(cond, ind, notfound), axis=axis)

def sum(nside, field, component, center=None, rmax=None, rmin=None):
  if center is None:
    center = field.cut.center
  if rmax is None:
    rmax = field.cut.size[0] / 2
  if rmin is None:
    rmin = rmax / 200

  disp = field['locations'] - center[newaxis, :]
  r = (disp[:,0:3] **2).sum(axis=-1) ** 0.5
  mask = (r < rmax) & (r > rmin)
  npix = hy.nside2npix(nside)
  pixid = hy.vec2pix(nside, disp[mask,0]/r[mask], disp[mask, 1]/r[mask], disp[mask,2]/r[mask])

  Mhist = zeros(npix)

  Mcount = bincount(pixid, field[component][mask])
  a = arange(len(Mcount))
  Mhist.reshape(-1)[a] = Mcount
  return Mhist

def Rs(nside, field, component, weight, levels=[0.1, 0.5, 0.9], center=None, rmax=None, rmin=None, direction=1, nbins=500, return_all=False):
  """ direction = +1 or -1, for increasing or decreasing """
  if center is None:
    center = field.cut.center
  if rmax is None:
    rmax = field.cut.size[0] / 2
  if rmin is None:
    rmin = rmax / 200

  disp = field['locations'] - center[newaxis, :]
  r = (disp[:,0:3] **2).sum(axis=-1) ** 0.5
  mask = (r < rmax) & (r > rmin)
  if nside > 0:
    npix = hy.nside2npix(nside)
    pixid = hy.vec2pix(nside, disp[mask,0]/r[mask], disp[mask, 1]/r[mask], disp[mask,2]/r[mask])
  else:
    npix = 1
    pixid = 0
  bins = linspace(rmin, rmax, nbins)
  rid = digitize(r[mask], bins)
  Mhist = zeros((npix , bins.shape[0]))
  HIhist = zeros_like(Mhist)

  flatid = pixid * bins.shape[0] + rid
  Mcount = bincount(flatid, field[weight][mask])
  HIcount = bincount(flatid, field[weight][mask] * field[component][mask])
  a = arange(len(Mcount))
  Mhist.reshape(-1)[a] = Mcount
  a = arange(len(HIcount))
  HIhist.reshape(-1)[a] = HIcount
  ahist = HIhist / Mhist

  result = []
  for l in levels:
    if direction > 0:
      ridout = first(ahist>l)
      ridin = last(ahist<l, first=ridout)
    else:
      ridout = first(ahist<l)
      ridin = last(ahist>l, first=ridout)

    rs = 0.5 * (bins[ridin] + bins[ridout + 1])

    result += [rs]
  if return_all:
    return result, ahist, bins
  else:
    return result

def HIIB(T):
  lIGMdaA = 2 * 157809 / T
  B = 2.753e-14 * lIGMdaA **1.500 / (1 + (lIGMdaA / 2.740) **0.407)**2.242
  return B

def analytic(lum, A, n_mean, clump, t, z):
  """returns rs(comoving kpc/h), t_rs(myear), r_ana"""
  rs = ((3 * lum) / (4 * 3.14 * A * clump * n_mean ** 2)) ** (1./3) * 3.24e-22 * (1 + z)
  t_rs = (A * clump * n_mean) **-1 / 3.14e7 / 1e6
  if t is not None:
    r_ana = (rs * (1 - exp(- t / t_rs))**(1.0/3.0))
  else:
    r_ana = None
  return rs, t_rs, r_ana

