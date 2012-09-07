import numpy
import SI
from units import Units
import _cosmology

def sphdist(ra1, dec1, ra2, dec2, out=None):
  """ all in radians """
  return _cosmology.sphdist(ra1, dec1, ra2, dec2, out)
def radec2vec(ra, dec):
  c = numpy.cos(dec)
  return numpy.asarray([c * numpy.cos(ra), c * numpy.sin(ra), numpy.sin(dec)]).T

def sphrotate(ra, dec, ref, new=(0, 0)):
  """ rotate ra dec, according to ref -> new, where ref=(ra, dec), new=(ra,dec)"""
  # modified from http://tir.astro.utoledo.edu/idl/spherical_coord_rotate.pro
  vref = radec2vec(*ref)
  vnew = radec2vec(*new)
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
  V = radec2vec(ra, dec).T
  V = ex.dot(V) * ex2[:, None] + ey.dot(V) * ey2[:, None] + ez.dot(V) * ez[:, None]
  dec = numpy.arcsin(V[2])
  ra = numpy.arctan2(V[1], V[0])
  ra[ra < 0] += 2 * numpy.pi
  return ra, dec

class Cosmology(object):
  cache = {}
  def __new__(klass, h, M, L, K=0):
    key = repr((h, M, L, K))
    if len(klass.cache) > 10: 
      klass.cache.clear()
    if key in klass.cache:
      return klass.cache[key]

    self = object.__new__(klass)
    self.M = M
    self.L = L
    self.K = K
    self.h = h
    self.units = Units(h)
    self.DH = self.units.C / self.units.H0
    self.tH = 1.0 / self.units.H0
    self._cosmology = _cosmology.Cosmology(self.M, self.K, self.L, self.h)

    z = numpy.linspace(0, 1, 1024*1024)
    z = z ** 6 * 255
    Ezinv = 1 / self.Ez(z)
    dz = numpy.zeros_like(z)
    dz[:-1] = numpy.diff(z)
    self._intEzinv_table = (Ezinv * dz).cumsum()[::256].copy()
    self._z_table = z[::256].copy()
    self._Ezinv_table = Ezinv[::256].copy()
    klass.cache[key] = self
    return self

  @classmethod
  def from_snapshot(cls, snapshot):
    if not 'OmegaM' in snapshot.C or not 'OmegaL' in snapshot.C or not 'h' in snapshot.C:
      warn("OmegaM, OmegaL, h not supported in snapshot, a default cosmology is used")
      return WMAP7
    else:
      return cls(K=0, M=snapshot.C['OmegaM'], L=snapshot.C['OmegaL'], h=snapshot.C['h'])
    
  def to_snapshot(self, snapshot):
    if not 'OmegaM' in snapshot.C or not 'OmegaL' in snapshot.C or not 'h' in snapshot.C:
      warn("OmegaM, OmegaL, h not supported in snapshot, cosmology not saved!")
      return
    snapshot.C['OmegaM'] = self.cosmology.M
    snapshot.C['OmegaL'] = self.cosmology.L
    snapshot.C['h'] = self.cosmology.h
    
  def __repr__(self):
    return "Cosmology(h=%g, M=%g, L=%g, K=%g)" % (self.h, self.M, self.L, self.K)

  def Ez(self, z, out=None):
   # Ez is dimensionless, and so is _cosmology
   return self._cosmology.eval('H', 1.0 / (1+ z), out)

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
    M,L,K = self.M, self.L, self.K
    if K!=0: raise ValueError("K has to be zero")
    aeq = (M / L) ** (1.0/3.0)
    pre = 2.0 / (3.0 * numpy.sqrt(L))
    arg = (a / aeq) ** (3.0 / 2.0) + numpy.sqrt(1.0 + (a / aeq) ** 3.0)
    return pre * numpy.log(arg) * self.tH

  def t2a(self, t):
    """ returns the scaling factor of the universe at given time(in GADGET unit)"""
    M,L,K = self.M, self.L, self.K
    if K!=0: raise ValueError("K has to be zero")
    aeq = (M / L) ** (1.0/3.0)
    pre = 2.0 / (3.0 * numpy.sqrt(L))
    return (numpy.sinh(t/self.tH/ pre)) ** (2.0/3.0) * aeq

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

  def ddplus(self, a, out=None):
    return self._cosmology.eval('ddplus', a, out)

  def dladt(self, a, out=None):
    """===============================================================================
    !  Evaluate dln(a)/dtau for FLRW cosmology.
    !===============================================================================
    """
    return self._cosmology.eval('dladt', a, out)

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
      return _cosmology.thirdleg(D1, D2, t, out)

  def D2z(self, z0, d):
    """returns the z satisfying Dc(z0, z) = d, and z > z0"""
    d0 = numpy.interp(z0, self._z_table, self._intEzinv_table)
    z = numpy.interp((d * (1 / self.DH) + d0), self._intEzinv_table, self._z_table)
    return z

  def radec2pos(self, ra, dec, z=None, Dc=None, out=None):
    """ only for flat cosmology, comoving coordinate is returned as (-1, 3) 
    """
    if Dc is None: Dc = self.Dc(z)
    return _cosmology.radec2pos(ra, dec, Dc, out)

  def pos2radec(self, pos, out=None):
    """ only for flat cosmology, return RA, DEC, Dc
    """
    return _cosmology.pos2radec(pos, out)

  def sphdist(self, ra1, dec1, ra2, dec2, out=None):
    """ all in rad 
       out cannot be alias of dec1, dec2 """
    return _cosmology.sphdist(ra1, dec1, ra2, dec2, out)

  def H(self, a, out=None):
    """ return the hubble constant at the given z or a, 
        in GADGET units,(and h is not multiplied)
    """
    out = self._cosmology.eval('H', a, out)
    out *= self.units.H0
    return out

  def DtoZ(self, distance, z0):
    """ Use D2z. integrate the redshift on a sightline based on the distance taking GADGET, comoving. 
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

  def ie2P(self, Xh, ie, ye, mass, out=None):
    """ from gadget internal energy per mass and density to 
       pressure integrated per particle volume(not the pressure)"""
    GAMMA = 5 / 3.
    mu = 1.0 / (ye * Xh + (1 - Xh) * 0.25 + Xh)
    return ie * mass * (Xh * (GAMMA - 1)) * ye * mu

  def ie2T(self, Xh, ie, ye, out=None):
    """ converts GADGET internal energy per mass to temperature. taking GADGET return GADGET.
       multiply by units.TEMPERATURE to SI"""
    fac = self.units.PROTONMASS / self.units.BOLTZMANN
    GAMMA = 5 / 3.
    mu = 1.0 / (ye * Xh + (1 - Xh) * 0.25 + Xh)
    if out != None:
      out[:] = ie[:] * mu * (GAMMA - 1) * fac
      return out
    else:
      return ie / (ye * Xh + (1 - Xh) * 0.25 + Xh) * (2.0 / 3.0) * fac

WMAP7 = Cosmology(K=0.0, M=0.28, L=0.72, h=0.72)
default = WMAP7
