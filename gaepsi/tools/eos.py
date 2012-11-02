from numpy import loadtxt
from numpy import interp

eos = None

def use(*args, **kwargs):
  global eos
  eos = loadtxt(*args, **kwargs)
  eos[:, 0] = 10**eos[:, 0]

def IGMfraction(rho, redshift):
  global eos
  return (1.0 - interp(rho, eos[:, 0]/ ((1. + redshift) **3), eos[:, 2], left=0.0, right=1.0))

