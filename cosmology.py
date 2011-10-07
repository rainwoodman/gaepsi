from numpy import trapz
from numpy import sqrt
from numpy import zeros_like
from numpy import diff
from numpy import pi
from numpy import log10
from numpy import log
from numpy import sinh
from numpy import linspace
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
    MYEAR_h = SI.MYEAR / TIME / h
    KPC = SI.KPC / LENGTH
    METER = SI.METER / LENGTH
    NANOMETER = SI.NANOMETER / LENGTH
    MYEAR = SI.MYEAR / TIME
    LYMAN_ALPHA_CROSSSECTION = SI.LYMAN_ALPHA_CROSSSECTION / (LENGTH**2)
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
  def __init__(self, OmegaR, OmegaM, OmegaL, h):
    self.Omega = dict(R = OmegaR, M = OmegaM, L = OmegaL)
    self.h = h
    self.units = Units(h)
  def a2t(self, a):
    """ returns the age of the universe at scaling factor a, in GADGET unit 
        (multiply by units.TIME to seconds, or divide by units.MYEAR_h to conv units)"""
    O = self.Omega
    H0 = self.units.H0
    aeq = (O['M'] / O['L']) ** (1.0/3.0)
    pre = 2.0 / (3.0 * sqrt(O['L']))
    arg = (a / aeq) ** (3.0 / 2.0) + sqrt(1.0 + (a / aeq) ** 3.0)
    return pre * log(arg) / H0
  def t2a(self, t):
    """ returns the scaling factor of the universe at given time(in GADGET unit)"""
    O = self.Omega
    H0 = self.units.H0
    aeq = (O['M'] / O['L']) ** (1.0/3.0)
    pre = 2.0 / (3.0 * sqrt(O['L']))
    return (sinh(self.units.H0 * t/ pre)) ** (2.0/3.0) * aeq

  def H(self, a):
    """ return the hubble constant at the given z or a, 
        in GADGET units,(and h is not multiplied)
    """
    O = self.Omega
  ## OmegaR != 0 this fails! 
    if not O['R']== 0: raise ValueError("OmegaR !=0 is not implemented")
    Omega0 = O['R'] + O['M'] + O['L']
    return self.units.H0 * sqrt(Omega0 / a**3 + (1 - Omega0 - O['L'])/ a**2 + O['L'])

  def DtoZ(self, distance, z0):
    """ integrate the redshift on a sightline based on the distance taking GADGET, comoving. 
        REF transform 3.38) in Barbara Ryden. """
    z = zeros_like(distance)
    dd = diff(distance)
    z[0] = z0
    for i in range(z.size - 1):
      z[i+1] = z[i] - self.H(1.0 / (z[i] + 1)) / C * dd[i]
    return z

  def Rvir(self, M, z, Deltac=200):
    """returns the virial radius at redshift z. taking GADGET, return GADGET, comoving.
       REF Rennna Barkana astro-ph/0010468v3 eq (24) [proper in eq 24]"""
    O = self.Omega
    OmegaMz = O['M'] * (1 + z)**3 / (O['M'] * (1+z)**3 + O['L'] + O['R'] * (1+z)**2)
    return 0.784 * (M * 100)**(0.33333) * (O['M'] / OmegaMz * Deltac / (18 * pi * pi))**-0.3333333 * 10

  def Vvir(self, M, z, Deltac=200):
    """ returns the physical virial circular velocity"""
    return (G * M / self.Rvir(M, z, Deltac) * (1.0 + z)) ** 0.5

  def Tvir(self, M, z, Deltac=200, Xh=0.76, ye=1.16):
    return 0.5 * self.Vvir(M,z, Deltac) ** 2 / (ye * Xh + (1 - Xh) * 0.25 + Xh)

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
    """ Hopkins etal 2006, added UV (13.6ev -> 250.0 ev with index=1.76),
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

default = Cosmology(OmegaR=0.0, OmegaM=0.255, OmegaL=0.745, h=0.702)

