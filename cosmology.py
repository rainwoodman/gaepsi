import numpy
from constant import SI
from units import Units

import cython._cosmology

def sphdist(ra1, dec1, ra2, dec2, out=None):
  """ all in radians """
  return cython._cosmology.sphdist(ra1, dec1, ra2, dec2, out)
def radec2vec(ra, dec):
  c = numpy.cos(dec)
  return numpy.asarray([c * numpy.cos(ra), c * numpy.sin(ra), numpy.sin(dec)])

def sphrotate(ref, new, ra, dec):
  """ rotate ra dec, according to ref -> new, where ref=(ra, dec), new=(ra,dec)"""
  # modified from http://tir.astro.utoledo.edu/idl/spherical_coord_rotate.pro
  vref = radec2vec(*ref)
  vnew = radec2vec(*new)
  X,Y,Z = radec2vec(ra, dec)
  ex = vref
  # old frame, ex -> ref, ez -> rotation axis
  ez = numpy.cross(vnew, vref)
  ez /= ez.dot(ez) ** 0.5
  ey = numpy.cross(ez, ex)
  ey /= ey.dot(ey) ** 0.5
  # new frame, ex2 -> new
  ex2 = vnew
  ey2 = numpy.cross(ez, ex2)
  ey2 /= ey2.dot(ey2) ** 0.5
  V = numpy.array([X, Y, Z])
  V = ex.dot(V) * ex2[:, None] + ey.dot(V) * ey2[:, None] + ez.dot(V) * ez[:, None]
  dec = numpy.arcsin(V[2])
  ra = numpy.arctan2(V[1], V[0])
  ra[ra < 0] += 2 * numpy.pi
  return ra, dec

class Cosmology:
  def __init__(self, h, M, L, K=0):
    self.M = M
    self.L = L
    self.K = K
    self.h = h
    self.units = Units(h)
    z = numpy.linspace(0, 1, 1024*1024)
    z = z ** 6 * 255
    Ezinv = 1 / self.Ez(z)
    dz = numpy.zeros_like(z)
    dz[:-1] = numpy.diff(z)
    self._intEzinv_table = (Ezinv * dz).cumsum()[::256]
    self._z_table = z[::256]
    self._Ezinv_table = Ezinv[::256]
    self.DH = self.units.C / self.units.H0

  def __repr__(self):
    return "Cosmology(h=%g M=%g, L=%g, K=%g)" % (self.h, self.M, self.L, self.K)

  def Ez(self, z):
    M,L,K = self.M, self.L, self.K
    return numpy.sqrt(M*(1+z)**3 + K*(1+z)**2+L)

  def intEzinvdz(self, z1, z2, func):
    """integrate func(z) * 1 /Ez dz, always from the smaller z to the higher"""
    if z1 < z2:
      mask = (self._z_table >= z1) & (self._z_table <= z2)
    else:
      mask = (self._z_table >= z2) & (self._z_table <= z1)
    z = self._z_table[mask]
    Ezinv = self._Ezinv_table[mask]
    v = func(z)
    return numpy.trapz(v * Ezinv, z)
  def intdV(self, z1, z2, func):
    return self.intEzinvdz(z1, z2, lambda z: self.Dc(z) ** 2 * 4 * numpy.pi * self.DH * func(z))

  def Vsky(self, z1, z2):
    return self.intdV(z1, z2, lambda z: 1.0)

  def a2t(self, a):
    """ returns the age of the universe at scaling factor a, in GADGET unit 
        (multiply by units.TIME to seconds, or divide by units.MYEAR_h to conv units)"""
    H0 = self.units.H0
    M,L,K = self.M, self.L, self.K
    if K!=0: raise ValueError("K has to be zero")
    aeq = (M / L) ** (1.0/3.0)
    pre = 2.0 / (3.0 * numpy.sqrt(L))
    arg = (a / aeq) ** (3.0 / 2.0) + numpy.sqrt(1.0 + (a / aeq) ** 3.0)
    return pre * numpy.log(arg) / H0

  def t2a(self, t):
    """ returns the scaling factor of the universe at given time(in GADGET unit)"""
    H0 = self.units.H0
    M,L,K = self.M, self.L, self.K
    if K!=0: raise ValueError("K has to be zero")
    aeq = (M / L) ** (1.0/3.0)
    pre = 2.0 / (3.0 * numpy.sqrt(L))
    return (numpy.sinh(H0 * t/ pre)) ** (2.0/3.0) * aeq

  def z2t(self, z):
    return self.a2t(1.0 / (1.0 + z))

  def t2z(self, t):
    return 1 / self.t2a(t) - 1

  def fomega(self, a=None, z=None):
    """!===============================================================================
       !  Evaluate f := dlog[D+]/dlog[a] (logarithmic linear growth rate) for
       !  lambda+matter-dominated cosmology.
       !  Omega0 := Omega today (a=1) in matter only.  Omega_lambda = 1 - Omega0.
       !===============================================================================
    """
    if a is None: a = 1 / (1.+z)
    eta = (self.M / a + self.L * a ** 2 + 1 - self.M - self.L) ** 0.5
    return (2.5 / self.dplus(a) - 1.5 * self.M / a + 1 - self.M - self.L) * eta **-2

  def dplus(self, a=None, z=None):
    """!===============================================================================
    !  Evaluate D+(a) (linear growth factor) for FLRW cosmology.
    !  Omegam := Omega today (a=1) in matter.
    !  Omegav := Omega today (a=1) in vacuum energy.
    !===============================================================================
    """
    if a is None: a = 1 / (1.+z)
    eta = (self.M / a + self.L * a ** 2 + 1 - self.M - self.L) ** 0.5
    agrid = numpy.linspace(1e-8, a, 100000)
    data = self.ddplus(agrid)
    return eta / a * numpy.trapz(y=data, x=agrid)

  def ddplus(self, a):
    eta = (self.M / a + self.L * a ** 2 + 1 - self.M - self.L) ** 0.5
    return 2.5 / eta ** 3

  def dladt(self, a):
    """===============================================================================
    !  Evaluate dln(a)/dtau for FLRW cosmology.
    !===============================================================================
    """
    eta = (self.M / a + self.L * a ** 2 + 1 - self.M - self.L) ** 0.5
    return a * eta

  def Dc(self, z1, z2=None, t=None, out=None):
    """ returns the comoing distance between z1 and z2 
        along two light of sights of angle t. 
        out cannot be an alias of t. """

    shape = numpy.broadcast(z1, z2, t).shape
    DH = self.DH
    if z2 is None: 
      z2 = z1
      z1 = 0
    D1 = numpy.interp(z1, self._z_table, self._intEzinv_table)
    D2 = numpy.interp(z2, self._z_table, self._intEzinv_table)
    if t is None:
      out = numpy.subtract(D2, D1, out)
      out *= DH
      return out
    else:
      D1 *= DH
      D2 *= DH
      return cython._cosmology.thirdleg(D1, D2, t, out)

  def D2z(self, z0, d):
    """returns the z satisfying Dc(z0, z) = d, and z > z0"""
    d0 = numpy.interp(z0, self._z_table, self._intEzinv_table)
    z = numpy.interp((d * (1 / self.DH) + d0), self._intEzinv_table, self._z_table)
    return z

  def radec2pos(self, ra, dec, z, out=None):
    """ only for flat cosmology, comoving coordinate is returned as (-1, 3) 
        ra cannot be an alias of out[:, 0], out[:, 2]
        dec cannot be an alias of out[:, 0]
        
    """
    return cython._cosmology.radec2pos(self, ra, dec, z, out)

  def sphdist(self, ra1, dec1, ra2, dec2, out=None):
    """ all in rad 
       out cannot be alias of dec1, dec2 """
    return cython._cosmology.sphdist(ra1, dec1, ra2, dec2, out)

  def H(self, a):
    """ return the hubble constant at the given z or a, 
        in GADGET units,(and h is not multiplied)
    """
    M,L,K = self.M, self.L, self.K
    H0 = self.units.H0
    Omega0 = K + M + L
    return H0 * numpy.sqrt(Omega0 / a**3 + (1 - Omega0 - L)/ a**2 + L)

  def DtoZ(self, distance, z0):
    """ integrate the redshift on a sightline based on the distance taking GADGET, comoving. 
        REF transform 3.38) in Barbara Ryden. """
    z = numpy.zeros_like(distance)
    dd =numpy.diff(distance)
    z[0] = z0
    for i in range(z.size - 1):
      z[i+1] = z[i] - self.H(1.0 / (z[i] + 1)) / self.units.C * dd[i]
    return z

  def Rvir(self, m, z, Deltac=200):
    """returns the virial radius at redshift z. taking GADGET, return GADGET, comoving.
       REF Rennna Barkana astro-ph/0010468v3 eq (24) [proper in eq 24]"""
    M,L,K = self.M, self.L, self.K
    OmegaMz = M * (1 + z)**3 / self.Ez(z)
    return 0.784 * (m * 100)**(0.33333) * (M / OmegaMz * Deltac / (18 * numpy.pi * numpy.pi))**-0.3333333 * 10

  def Vvir(self, m, z, Deltac=200):
    """ returns the physical virial circular velocity"""
    return (G * M / self.Rvir(m, z, Deltac) * (1.0 + z)) ** 0.5

  def Tvir(self, m, z, Deltac=200, Xh=0.76, ye=1.16):
    return 0.5 * self.Vvir(m,z, Deltac) ** 2 / (ye * Xh + (1 - Xh) * 0.25 + Xh)

  def ie2T(self, Xh, ie, ye, out=None):
    """ converts GADGET internal energy per mass to temperature. taking GADGET return GADGET.
       multiply by units.TEMPERATURE to SI"""
    fac = self.units.PROTONMASS / self.units.BOLTZMANN
    if out != None:
      out[:] = ie[:] / (ye[:] * Xh + (1 - Xh) * 0.25 + Xh) * (2.0 / 3.0) * fac
      return out
    else:
      return ie / (ye * Xh + (1 - Xh) * 0.25 + Xh) * (2.0 / 3.0) * fac

  def Lblue(self, mdot):
    """ converts GADGET bh accretion rate to Blue band bolemetric luminosity taking GADGET return GADGET,
        multiply by units.POWER to SI """
    def f(x): return 0.80 - 0.067 * (numpy.log10(x) - 12) + 0.017 * (numpy.log10(x) - 12)**2 - 0.0023 * (numpy.log10(x) - 12)**3
    # 0.1 is coded in gadget bh model.
    L = mdot * self.units.C ** 2 * 0.1
    return 10**(-f(L/self.units.SOLARLUMINOSITY)) * L
  def Lsoft(self, mdot):
    """ converts GADGET bh accretion rate to Blue band bolemetric luminosity taking GADGET return GADGET,
        multiply by units.POWER to SI """
    def f(x): return 1.65 + 0.22 * (numpy.log10(x) - 12) + 0.012 * (numpy.log10(x) - 12)**2 - 0.0015 * (numpy.log10(x) - 12)**3
    # 0.1 is coded in gadget bh model.
    L = mdot * self.units.C ** 2 * 0.1
    return 10**(-f(L/self.units.SOLARLUMINOSITY)) * L

  def QSObol(self, mdot, type):
    """ Hopkins etal 2007, added UV (13.6ev -> 250.0 ev with index=1.76),
        type can be 'blue', 'ir', 'soft', 'hard', or 'uv'. """
    params = {
      'blue': (6.25,-0.37,9.00,-0.012,),
      'ir':   (7.40,-0.37, 10.66,-0.014,),
      'soft':(17.87,0.28,10.03,-0.020,),
      'hard':(10.83,0.28,6.08,-0.020,),
    }

    L = mdot * self.units.C ** 2 * 0.1
    if type == 'bol': return L 

    if type == 'uv': 
      c1,k1,c2,k2 = params['blue']
    else:
      c1,k1,c2,k2 = params[type]
    x = L / (1e10 * self.units.SOLARLUMINOSITY)
    ratio = c1 * x ** k1 + c2 * x ** k2
    if type == 'uv':
      LB = L / ratio
      fB = self.units.C / (445e-9 * self.units.NANOMETER)
      fX = self.units.C / (120e-9 * self.units.NANOMETER)
      fI = self.units.C / (91.1e-9 * self.units.NANOMETER)
      lB = LB / fB
      lX = lB * (fX / fB) ** -0.44
      lI = lX * (fI / fX) ** -1.76
      Lband = lI * fI / -0.76 * ((250.0 / 13.6) ** -0.76 - 1)
    else:
      Lband = L / ratio
    return Lband

WMAP7 = Cosmology(K=0.0, M=0.28, L=0.72, h=0.72)

