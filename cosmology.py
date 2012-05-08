from numpy import ndarray
from numpy import trapz
from numpy import sqrt
from numpy import zeros_like
from numpy import empty_like
from numpy import empty
from numpy import diff
from numpy import pi
from numpy import log10
from numpy import log
from numpy import sin, cos
from numpy import sinh
from numpy import linspace
from numpy import logspace
from numpy import interp
from numpy import cumsum, zeros
from numpy import arccos
from numpy import clip
from numpy import subtract
from numpy import multiply
from numpy import add
from numpy import broadcast

from constant import SI

class Units:
  def __init__(self, h):
    C = 3e5
    G = 43007.1
    H0 = 0.1
    pi = 3.1415926
    #internal units
    TIME = H0 / (SI.H0 * h)
    LENGTH = TIME * (SI.C / C)
    MASS = G / SI.G * LENGTH ** 3 * TIME ** -2
    TEMPERATURE = 1.0
    ENERGY = MASS * (LENGTH / TIME) ** 2
    POWER = ENERGY / TIME

    #quantities
    BOLTZMANN = SI.BOLTZMANN / ((LENGTH / TIME) ** 2 * MASS)
    SOLARMASS = SI.SOLARMASS / MASS
    SOLARLUMINOSITY = SI.SOLARLUMINOSITY / POWER
    PROTONMASS = SI.PROTONMASS / MASS
    KPC_h = SI.KPC / LENGTH / h
    MPC_h = KPC_h * 1000
    MYEAR_h = SI.MYEAR / TIME / h
    KPC = SI.KPC / LENGTH
    MPC = KPC * 1000
    METER = SI.METER / LENGTH
    NANOMETER = SI.NANOMETER / LENGTH
    MYEAR = SI.MYEAR / TIME
    LYMAN_ALPHA_CROSSSECTION = SI.LYMAN_ALPHA_CROSSSECTION / (LENGTH**2)
    J = SI.J / ENERGY
    W = SI.W / (ENERGY / TIME)
    EV = SI.EV / ENERGY
    SECOND = SI.SECOND / TIME
    RYDBERG = SI.RYDBERG / ENERGY
    CRITICAL_DENSITY = 3 * H0 ** 2/ (8 * pi * G)

    self.set_dict(locals());

  def set_dict(self, locals):
    for field in locals:
      if field == "self": continue
      self.__dict__[field] = locals[field]
  def __str__(self):
    return str(self.__dict__)

class Cosmology:
  def __init__(self, h, M, L, K=0):
    self.M = M
    self.L = L
    self.K = K
    self.h = h
    self.units = Units(h)
    z = linspace(0, 1, 1024*1024)
    z = z ** 6 * 255
    Ezinv = 1 / self.Ez(z)
    dz = zeros_like(z)
    dz[:-1] = diff(z)
    self._intEzinv_table = (Ezinv * dz).cumsum()[::256]
    self._z_table = z[::256]
    self._Ezinv_table = Ezinv[::256]
  def __repr__(self):
    return "Cosmology(h=%g M=%g, L=%g, K=%g)" % (self.h, self.M, self.L, self.K)
  @property
  def DH(self):
    return self.units.C / self.units.H0

  def Ez(self, z):
    M,L,K = self.M, self.L, self.K
    return sqrt(M*(1+z)**3 + K*(1+z)**2+L)

  def intEzinvdz(self, z1, z2, func):
    """integrate func(z) * 1 /Ez dz, always from the smaller z to the higher"""
    if z1 < z2:
      mask = (self._z_table >= z1) & (self._z_table <= z2)
    else:
      mask = (self._z_table >= z2) & (self._z_table <= z1)
    z = self._z_table[mask]
    Ezinv = self._Ezinv_table[mask]
    v = func(z)
    return trapz(v * Ezinv, z)
  def intdV(self, z1, z2, func):
    return self.intEzinvdz(z1, z2, lambda z: self.Dc(z) ** 2 * 4 * pi * self.DH * func(z))

  def Vsky(self, z1, z2):
    return self.intdV(z1, z2, lambda z: 1.0)

  def a2t(self, a):
    """ returns the age of the universe at scaling factor a, in GADGET unit 
        (multiply by units.TIME to seconds, or divide by units.MYEAR_h to conv units)"""
    H0 = self.units.H0
    M,L,K = self.M, self.L, self.K
    if K!=0: raise ValueError("K has to be zero")
    aeq = (M / L) ** (1.0/3.0)
    pre = 2.0 / (3.0 * sqrt(L))
    arg = (a / aeq) ** (3.0 / 2.0) + sqrt(1.0 + (a / aeq) ** 3.0)
    return pre * log(arg) / H0

  def t2a(self, t):
    """ returns the scaling factor of the universe at given time(in GADGET unit)"""
    H0 = self.units.H0
    M,L,K = self.M, self.L, self.K
    if K!=0: raise ValueError("K has to be zero")
    aeq = (M / L) ** (1.0/3.0)
    pre = 2.0 / (3.0 * sqrt(L))
    return (sinh(H0 * t/ pre)) ** (2.0/3.0) * aeq

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
    agrid = linspace(1e-8, a, 100000)
    data = self.ddplus(agrid)
    return eta / a * trapz(y=data, x=agrid)

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

    shape = broadcast(z1, z2, t).shape
    if out is None:
      out = empty(shape=shape, dtype='f4')

    DH = self.DH
    if z2 is None: 
      z2 = z1
      z1 = 0
    D1 = interp(z1, self._z_table, self._intEzinv_table)
    D2 = interp(z2, self._z_table, self._intEzinv_table)
    if t is None:
      subtract(D2, D1, out)
      multiply(out, DH, out)
      return out
    else:
#
#     z1-..-.-.+ |D|
#    / )  DM1   \
#   /t  )        \
#  o----z1'--D21--z2
#   

      D21 = empty(shape=broadcast(D2, D1).shape, dtype='f4')
      subtract(D2, D1, D21)

#     DM1 = 2 * sin(t * 0.5) * D1
      
      del D2

      tmp = empty_like(t, dtype='f4')
      multiply(t , 0.5, tmp)
      sin(tmp, tmp)
      DM1 = empty(shape=broadcast(D1, t).shape, dtype='f4')
      multiply(D1, tmp, DM1)
      multiply(DM1, 2, DM1)

      del D1
#      D = sqrt(DM1 **2 + D21 **2 - 2 * cos(0.5 * (pi + t)) * DM1 * D21)
      # cross term first
      add(t, pi, tmp)
      multiply(tmp, 0.5, tmp)
      cos(tmp, tmp)
      multiply(tmp, 2, tmp)
      multiply(tmp, DM1, out)
      multiply(out, D21, out)

      multiply(DM1, DM1, DM1)
      multiply(D21, D21, D21)

      subtract(DM1, out, out)
      add(out, D21, out)
      sqrt(out, out)
      multiply(out, DH, out)
      return out

  def D2z(self, z0, d):
    """returns the z satisfying Dc(z0, z) = d, and z > z0"""
    d0 = interp(z0, self._z_table, self._intEzinv_table)
    z = interp((d / self.DH + d0), self._intEzinv_table, self._z_table)
    return z

  def radec2pos(self, ra, dec, z, out=None):
    """ only for flat cosmology, comoving coordinate is returned as (-1, 3) 
        ra cannot be an alias of out[:, 0], out[:, 2]
        dec cannot be an alias of out[:, 0]
        
    """
    shape = broadcast(dec, ra, z).shape
    if out is None:
      out = empty(shape=shape, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    else:
      out = out.view(dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
      
    
#    out[:, 0] = r * cos(dec) * cos(ra)
#    out[:, 1] = r * cos(dec) * sin(ra)
#    out[:, 2] = r * sin(dec)

    tmp = self.Dc(z)
    cos(dec, out['x'])
    sin(dec, out['z'])
    multiply(out['x'], tmp, out['x'])
    multiply(out['z'], tmp, out['z'])
    del tmp
    tmp = cos(ra)
    multiply(out['x'], sin(ra), out['y'])
    multiply(out['x'], tmp, out['x'])

    return out.view(dtype=('f4', 3))

  def sphdist(self, ra1, dec1, ra2, dec2, out=None):
    """ all in rad 
       out cannot be alias of dec1, dec2 """
    shape = broadcast(ra1, dec1, ra2, dec2).shape
    if out is None:
      out = empty(shape=shape, dtype='f4')

    #cr = cos(ra1) * cos(ra2)
    multiply(cos(ra1), cos(ra2), out)

    t1 = empty(shape=broadcast(ra1, ra2).shape, dtype='f4')
    #sr = sin(ra1) * sin(ra2)
    multiply(sin(ra1), sin(ra2), t1)

    # (cr + sr)
    add(out, t1, out)
    del t1
    #cd = cos(dec1) * cos(dec2)
    t1 = empty(shape=broadcast(dec1, dec2).shape, dtype='f4')
    multiply(cos(dec1), cos(dec2), t1)
    
    # cd * (sr + cr)
    multiply(t1, out, out)

    # sd = sin(dec1) * sin(dec2)
    multiply(sin(dec1), sin(dec2), t1)

    # sd += cd * (sr + cr)
    add(t1, out, out)
   
    clip(out, a_min=-1, a_max=1, out=out)
    return arccos(out, out)

  def H(self, a):
    """ return the hubble constant at the given z or a, 
        in GADGET units,(and h is not multiplied)
    """
    M,L,K = self.M, self.L, self.K
    H0 = self.units.H0
    Omega0 = K + M + L
    return H0 * sqrt(Omega0 / a**3 + (1 - Omega0 - L)/ a**2 + L)

  def DtoZ(self, distance, z0):
    """ integrate the redshift on a sightline based on the distance taking GADGET, comoving. 
        REF transform 3.38) in Barbara Ryden. """
    z = zeros_like(distance)
    dd = diff(distance)
    z[0] = z0
    for i in range(z.size - 1):
      z[i+1] = z[i] - self.H(1.0 / (z[i] + 1)) / self.units.C * dd[i]
    return z

  def Rvir(self, m, z, Deltac=200):
    """returns the virial radius at redshift z. taking GADGET, return GADGET, comoving.
       REF Rennna Barkana astro-ph/0010468v3 eq (24) [proper in eq 24]"""
    M,L,K = self.M, self.L, self.K
    OmegaMz = M * (1 + z)**3 / self.Ez(z)
    return 0.784 * (m * 100)**(0.33333) * (M / OmegaMz * Deltac / (18 * pi * pi))**-0.3333333 * 10

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
    def f(x): return 0.80 - 0.067 * (log10(x) - 12) + 0.017 * (log10(x) - 12)**2 - 0.0023 * (log10(x) - 12)**3
    # 0.1 is coded in gadget bh model.
    L = mdot * self.units.C ** 2 * 0.1
    return 10**(-f(L/self.units.SOLARLUMINOSITY)) * L
  def Lsoft(self, mdot):
    """ converts GADGET bh accretion rate to Blue band bolemetric luminosity taking GADGET return GADGET,
        multiply by units.POWER to SI """
    def f(x): return 1.65 + 0.22 * (log10(x) - 12) + 0.012 * (log10(x) - 12)**2 - 0.0015 * (log10(x) - 12)**3
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

