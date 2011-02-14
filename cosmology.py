from scipy.integrate import quad
from numpy import sqrt
from numpy import zeros_like
from numpy import diff
from numpy import pi
from constant.GADGET import *

class Cosmology:
  def __init__(self, OmegaR, OmegaM, OmegaL, h):
    self.R = OmegaR
    self.M = OmegaM
    self.L = OmegaL
    self.h = h
  
  def age(self, z=None, a=None):
    """ returns the age of the universe at the given z or a, in SI"""
    if a==None: a = 1.0 / (1.0 + z)
    Omega0 = self.R + self.M + self.L
    f = lambda x: sqrt(1.0/(self.R/x**2 + self.M/x + self.L * x**2 + 1 - Omega0))
    return quad(f, 0, a)[0] / (H0 * h)

  def H(self, z=None, a=None):
    """ return the hubble constant at the given z or a, in GADGET units, h is not multiplied"""
  ## OmegaR != 0 this fails! 
    if not self.R == 0: raise ValueError("OmegaR !=0 is not implemented")
    if a==None: a = 1.0 / (1.0 + z)
    Omega0 = self.R + self.M + self.L
    return H0 * sqrt(Omega0 / a**3 + (1 - Omega0 - self.L)/ a**2 + self.L)

  def DtoZ(self, distance, z0):
    """ integrate the redshift on a sightline based on the distance taking GADGET, comoving. 
        REF transform 3.38) in Barbara Ryden. """
    z = zeros_like(distance)
    dd = diff(distance)
    z[0] = z0
    for i in range(z.size - 1):
      z[i+1] = z[i] - self.H(z[i]) / C * dd[i]
    return z

  def Rvir(self, M, z, Deltac=200):
    """returns the virial radius at redshift z. taking GADGET, return GADGET, comoving.
       REF Rennna Barkana astro-ph/0010468v3 eq (24) [proper in eq 24]"""
    OmegaMz = self.M * (1 + z)**3 / (self.M * (1+z)**3 + self.L + self.R * (1+z)**2)
    return 0.784 * (M * 100)**(0.33333) * (self.M / OmegaMz * Deltac / (18 * pi * pi))**-0.3333333 * 10

default = Cosmology(OmegaR=0.0, OmegaM=0.26, OmegaL=0.74, h=0.72)

