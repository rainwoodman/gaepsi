from scipy.integrate import quad
from numpy import sqrt
from numpy import zeros_like
from numpy import diff
from constant.SI import *

class Cosmology:
  def __init__(self, OmegaR, OmegaM, OmegaL, h):
    self.R = OmegaR
    self.M = OmegaM
    self.L = OmegaL
    self.h = h
  
  def time(self, z=None, a=None):
    if a==None: a = 1.0 / (1.0 + z)
    Omega0 = self.R + self.M + self.L
    f = lambda x: sqrt(1.0/(self.R/x**2 + self.M/x + self.L * x**2 + 1 - Omega0))
    return quad(f, 0, a)[0] / (H0 * h)

  def H(self, z=None, a=None):
  ## OmegaR != 0 this fails! 
    if not self.R == 0: raise ValueError("OmegaR !=0 is not implemented")
    if a==None: a = 1.0 / (1.0 + z)
    Omega0 = self.R + self.M + self.L
    return H0 * sqrt(Omega0 / a**3 + (1 - Omega0 - self.L)/ a**2 + self.L)

  def z_sightline(self, distance, z0):
    """ integrate the redshift on a sightline based on the distance taking SI"""
    z = zeros_like(distance)
    dd = diff(distance)
    z[0] = z0
    for i in range(z.size - 1):
      z[i+1] = z[i] - self.H(z[i]) / C * dd[i]
    return z
    
default = Cosmology(OmegaR=0.0, OmegaM=0.3, OmegaL=0.7, h=0.72)

