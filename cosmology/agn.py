import numpy
import sharedmem
from _qlf import qlf as _qlf
from gaepsi.cosmology import Cosmology

HOPKINS2007 = Cosmology(M=0.3, L=0.7, h=0.7)

_qlf_interp = {}

def _bandconv(band, cosmology=HOPKINS2007, hertz=False):
  U = cosmology.units
  if numpy.isscalar(band): return band
  unit = band[0]
  if not isinstance(unit, basestring):
    return numpy.array(band, dtype='f8')
  rt = numpy.array(band[1:], dtype='f8')
  if unit == 'A':
      rt *= U.ANGSTROM
  elif unit == 'ev':
      rt[:] = U.PLANCK * U.C / (U.EV * rt)
  if hertz: 
      rt[:] = U.C / rt / U.HERTZ
  rt.sort()

  if len(rt) > 2: raise ValueError("band %s not recognoized. use ('A', low, high) or ('ev', low, high) or a tuple" % band)
  if len(rt) == 2: return tuple(rt)
  return rt[0]

class QLFfunction:
  def __init__(self, func):
    self.func = func
  def __call__(self, z, L):
    """ returns the number density per internal volume per log10(L), z is redshifit, L is log10(L/Lsun)"""
    return self.func(z, L)
  def integral(self, z, L, epsilon=1e-3):
    """ returns integrated QLF.
        z, L can be scalar or a range (min, max).
        integration is performed from min to max if a range is given, along that direction.
        
        z is redshift, and L is in log10(L/Lsun).
        epsilon decides the small range used to determine the 1d integral.
    """
    if numpy.isscalar(z):
      return self.func.integral(z-epsilon, z+epsilon, L[0], L[1]) / epsilon * 0.5
    if numpy.isscalar(L):
      return self.func.integral(z[0], z[1], L-epsilon, L+epsilon) / epsilon * 0.5
    return self.func.integral(z[0], z[1], L[0], L[1])

def qlf(band, cosmology=HOPKINS2007):
  """ returns an scipy interpolated function for the hopkins 2007 QLF, 
      at a given band.
      band can be a frequency, or 'bol', 'blue', 'ir', 'soft', 'hard'.
      HOPKINS2007 cosmology is implied.
      result shall not depend on the input cosmology but the HOPKINS cosmology
      is implied in the fits.
  """
  U = cosmology.units
  banddict = {'bol':0, 'blue':-1, 'ir':-2, 'soft':-3, 'hard': -4}
  from scipy.interpolate import RectBivariateSpline
  if band in banddict: band = banddict[band]
  else: band = _bandconv(band, cosmology, hertz=True)
  if band not in _qlf_interp:
    Lbol = numpy.linspace(8, 18, 200)
    zrange = numpy.linspace(0, 6, 300)
    v = numpy.empty(numpy.broadcast(Lbol[None, :], zrange[:, None]).shape)
    Lband, M_AB, S_nu, Phi = _qlf(band, 1.0, Lbol)
    with sharedmem.Pool(use_threads=True) as pool:
      def work(v, z):
        Lband_, M_AB, S_nu, Phi = _qlf(band, z, Lbol)
        v[:] = Phi
      pool.starmap(work, zip(v, zrange))
    v /= (U.MPC ** 3)
    # Notice that hopkins used 3.9e33 ergs/s for Lsun, but we use a different number.
    # but the internal fits of his numbers
    # thus we skip the conversion, in order to match the fits with luminosity in terms of Lsun.
    # Lbol = Lbol - numpy.log10(U.SOLARLUMINOSITY/U.ERG*U.SECOND) + numpy.log10(3.9e33)
    func = RectBivariateSpline(zrange, Lband, v)
    _qlf_interp[band] = QLFfunction(func)
  return _qlf_interp[band]
  
def bhLbol(mdot, cosmology=HOPKINS2007):
  """ converts accretion rate in internal units to bolemetric, in units of Lsun """
  U = cosmology.units
  L = mdot * U.C ** 2 * 0.1
  return L / U.SOLARLUMINOSITY

def bolemetric(Lbol, band, cosmology=HOPKINS2007):
    """ 
        Lbol is in Lsun.
        Hopkins etal 2007, added UV (13.6ev -> 250.0 ev with index=1.76),
        type can be 'bol', 'blue', 'ir', 'soft', 'hard', 'uv'
        or one of the follows:
        - (min, max), in A,
        - ('A', min, max), in A,
        - ('ev', min, max), in ev,
        'uv' ::= ('ev', 13.6, 250.0)
        total band lumisoty is returned, and only from 50A to 1um is covered
        returns the bolemtric luminosity in Lsun.
        result shall not depend on the input cosmology but the HOPKINS cosmology
        is implied in the fits.
    """ 
    U = cosmology.units
    params = {
      'blue': (6.25,-0.37,9.00,-0.012,),
      'ir':   (7.40,-0.37, 10.66,-0.014,),
      'soft':(17.87,0.28,10.03,-0.020,),
      'hard':(10.83,0.28,6.08,-0.020,),
    }

    if band == 'uv':
      band = ('ev', 13.6, 250.)

    A1200 = 1200e-10 * U.METER
    A500 = 500e-10 * U.METER
    A50 = 50e-10 * U.METER   # about 250 ev
    A2500 = 2500e-10 * U.METER
    A4450 = 4450e-10 * U.METER
    A2k = U.C * U.PLANCK / (2000 * U.EV) # about 6.2 A

    if band not in params:
      c1,k1,c2,k2 = params['blue']
      # extrapolate from lue band luminosity
      Amin, Amax = _bandconv(band, cosmology)

      if Amin < 49e-10 * U.METER \
      or Amax > 1e-6 * U.METER:
        raise ValueError('input band %s(%g A, %g A) not implemented' % (band, Amin / U.METER * 1e10, Amax / U.METER * 1e10))
    else:
      c1,k1,c2,k2 = params[band]

    x = Lbol * 1e-10
    ratio = c1 * x ** k1 + c2 * x ** k2

    if band not in params:
      L4450 = Lbol / ratio
      l4450 = L4450 / (U.C / A4450)
      l1200 = l4450 * (A4450 / A1200) ** -0.44
      l500 = l1200 * (A1200 / A500) ** -1.76
      l2500 = l4450 * (A4450 / A2500) ** -0.44
      alphaOX = -0.107 * numpy.log10(l2500 * U.LSUN / U.ERG) + 1.739
      l2k = l2500 / 10 ** (alphaOX / -0.384)
      # OK OK 1.8 is the intrinsic photon index, so the spectra index shall be 1.8 + 1.
      # am I right?
      l50 = l2k * (A2k / A50) ** 2.8
      alpha50 = numpy.log10(l500 / l50) / numpy.log10(A50 / A500)
      def piece(Alow, Ahigh, alpha, lref, Aref):
        alpha = numpy.asarray(alpha)
        if (Amin > Ahigh) or (Amax < Alow): return 0
        Amin_, Amax_ = numpy.clip([Amin, Amax], Alow, Ahigh)
        lmax = lref * (Aref / Amax_) ** alpha
        return lmax / Amax_ * U.C / (alpha + 1) * ((Amax_ / Amin_) ** (alpha + 1) - 1)

      res = [
            # this piece bridges UV and X ray
            # we assume -1.76 but it seems to be between -1.76 and alphaOX,
            # very complicated stuff. GABE's old code seems to be assuming -1.76, too.
             piece(A50, A500, alpha50, l50, A50), 
            # UV piece
             piece(A500, A1200, -1.76, l1200, A1200),
            # Optical piece
             piece(A1200, A4450 * 10, -0.44, l4450, A4450)]

      print res
      return reduce(numpy.add,res)
    else:
      Lband = Lbol / ratio
    return Lband

if False:
  def QSObol(U, mdot, type):
    """ this code is incorrectly handling 50-500A.
       Hopkins etal 2007, added UV (13.6ev -> 250.0 ev with index=1.76),
type can be 'blue', 'ir', 'soft', 'hard', or 'uv'. """
    warn("QSObol is deprecated it incorrectly handles 50-500A")
    params = {
      'blue': (6.25,-0.37,9.00,-0.012,),
      'ir': (7.40,-0.37, 10.66,-0.014,),
      'soft':(17.87,0.28,10.03,-0.020,),
      'hard':(10.83,0.28,6.08,-0.020,),
    }

    L = mdot * U.C ** 2 * 0.1
    if type == 'bol': return L

    if type == 'uv':
      c1,k1,c2,k2 = params['blue']
    else:
      c1,k1,c2,k2 = params[type]
    x = L / (1e10 * U.SOLARLUMINOSITY)
    ratio = c1 * x ** k1 + c2 * x ** k2
    if type == 'uv':
      LB = L / ratio
      fB = U.C / (445e-9 * U.METER)
      fX = U.C / (120e-9 * U.METER)
      fI = U.C / (91.1e-9 * U.METER)
      lB = LB / fB
      lX = lB * (fX / fB) ** -0.44
      lI = lX * (fI / fX) ** -1.76
      print 'lI = ', lI
      Lband = lI * fI / -0.76 * ((250.0 / 13.6) ** -0.76 - 1)
    else:
      Lband = L / ratio
    return Lband
  def Lblue(self, mdot):
    """ converts GADGET bh accretion rate to Blue band bolemetric luminosity taking GADGET return GADGET,
        multiply by U.POWER to SI """
    def f(x): return 0.80 - 0.067 * (numpy.log10(x) - 12) + 0.017 * (numpy.log10(x) - 12)**2 - 0.0023 * (numpy.log10(x) - 12)**3
    # 0.1 is coded in gadget bh model.
    L = mdot * self.U.C ** 2 * 0.1
    return 10**(-f(L/self.U.SOLARLUMINOSITY)) * L
  def Lsoft(self, mdot):
    """ converts GADGET bh accretion rate to Blue band bolemetric luminosity taking GADGET return GADGET,
        multiply by U.POWER to SI """
    def f(x): return 1.65 + 0.22 * (numpy.log10(x) - 12) + 0.012 * (numpy.log10(x) - 12)**2 - 0.0015 * (numpy.log10(x) - 12)**3
    # 0.1 is coded in gadget bh model.
    L = mdot * self.U.C ** 2 * 0.1
    return 10**(-f(L/self.U.SOLARLUMINOSITY)) * L
