import numpy
import sharedmem
import _qlf

from gaepsi.cosmology import Cosmology

HOPKINS2007 = Cosmology(M=0.3, L=0.7, h=0.7)

_qlf_interp = {}

def _bandconv(U, band, hertz=False):
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

bands = lambda x:0
bands.u = ('A', 3541)
bands.g = ('A', 4653)
bands.b = ('A', 4450)
bands.r = ('A', 6147)
bands.i = ('A', 7461)
bands.i2 = ('A', 2500) # z=2 iband used in SDSS
bands.z = ('A', 8904)

class QLFfunction:
  def __init__(self, func, magnitude):
    self.func = func
    self.magnitude = magnitude
  def __call__(self, z, L):
    """ returns the number density per internal volume per log10(L), z is redshifit, L is log10(L/Lsun), input must be on a grid, and for Magnitude must be decreasing"""
    if self.magnitude :
      return self.func(z, (-L))
    else:
      return self.func(z, L)
    raise
  def integral(self, z, L, epsilon=1e-3):
    """ returns integrated QLF.
        z, L can be scalar or a range (min, max).
        integration is performed from min to max if a range is given, along that direction.
        
        z is redshift, and L is in log10(L/Lsun).
        epsilon decides the small range used to determine the 1d integral.
    """
    if numpy.isscalar(z):
      if self.magnitude:
        L = (-L[0], -L[1])
      return self.func.integral(z-epsilon, z+epsilon, L[0], L[1]) / epsilon * 0.5
    if numpy.isscalar(L):
      if self.magnitude:
        L = -l
      return self.func.integral(z[0], z[1], L-epsilon, L+epsilon) / epsilon * 0.5
    if self.magnitude:
      L = (-L[0], -L[1])
    return self.func.integral(z[0], z[1], L[0], L[1])

def qlf_observed(cosmology, band=bands.i, faint=21.8, bright=17.8):
    """returns the integrate QLF as function of z, integrated from 
       apparent magnitude at the observer band given in am_limit,
       FIXME: no K-correction is done """
    U = cosmology.U
    from scipy.interpolate import interp1d, \
            InterpolatedUnivariateSpline, \
            UnivariateSpline
    band = _bandconv(U, band, hertz=True)
    # now band is in herts, local frame.
    zbins = numpy.linspace(0.0, 6.0, 31, endpoint=True)
    zbins[0] = 0.1
    Lbolbins=numpy.linspace(11, 16, 100) # hope this will cover everything.

    def work(z):
        Lband, M_AB, S_nu, Phi = _qlf.qlf(band, z, Lbolbins)
        Phi /= (U.MPC ** 3)
        Dc = cosmology.Dc(z)
        DL = Dc * (1 + z)
        distance_modulus = 5 * numpy.log10(DL / (0.01 * U.KPC))
        m_AB = M_AB + distance_modulus
        spl = UnivariateSpline(-m_AB, Phi, k=5)
        print z, DL, m_AB[0], m_AB[-1], Phi.sum(), m_AB.max(), m_AB.min()
        integrated = spl.integral(-faint, -bright)
        if integrated < 0: integrated = 0
        return integrated
    with sharedmem.Pool() as pool:
        integrated = numpy.array(pool.map(work, zbins))
    print zbins, integrated
    return interp1d(zbins, integrated, bounds_error=False, fill_value=0)

def qlf(cosmology, band, magnitude=False, returnraw=False):
  """ returns an scipy interpolated function for the hopkins 2007 QLF, 
      at a given band.
      band can be a frequency, or 'bol', 'blue', 'ir', 'soft', 'hard'.
      HOPKINS2007 cosmology is implied.
      result shall not depend on the input cosmology but the HOPKINS cosmology
      is implied in the fits.
      if return raw is true, return 
         zrange, Lbol, Lband, M_AB, S_nu, Phi
  """
  zbins = numpy.linspace(0, 6, 200)
  Lbolbins=numpy.linspace(8, 18, 300)

  U = cosmology.units
  banddict = {'bol':0, 'blue':-1, 'ir':-2, 'soft':-3, 'hard': -4}
  from scipy.interpolate import RectBivariateSpline
  if band in banddict: 
      band = banddict[band]

  else: band = _bandconv(U, band, hertz=True)
  key = band

  if key not in _qlf_interp:
    v = numpy.empty(numpy.broadcast(Lbolbins[None, :], zbins[:, None]).shape)
    Lband, M_AB, S_nu, Phi = _qlf.qlf(band, 1.0, Lbolbins)
    with sharedmem.TPool() as pool:
      def work(v, z):
        Lband_, M_AB_, S_nu, Phi = _qlf.qlf(band, z, Lbolbins)
        v[:] = Phi
      pool.starmap(work, zip(v, zbins))
    v /= (U.MPC ** 3)
    # Notice that hopkins used 3.9e33 ergs/s for Lsun, but we use a different number.
    # but the internal fits of his numbers
    # thus we skip the conversion, in order to match the fits with luminosity in terms of Lsun.
    # Lbol = Lbol - numpy.log10(U.SOLARLUMINOSITY/U.ERG*U.SECOND) + numpy.log10(3.9e33)
    data = numpy.empty(shape=len(Lbolbins),
        dtype=[
          ('Lbol', 'f4'),
          ('Lband', 'f4'),
          ('M_AB', 'f4'),
          ('S_nu', 'f4'),
          ('Phi', ('f4', v.shape[0]))])
    data['Lbol'] = Lbolbins
    data['Lband'] = Lband
    data['M_AB'] = M_AB
    data['S_nu'] = S_nu
    data['Phi'] = v.T
    _qlf_interp[key] = data
  data = _qlf_interp[key]
  if returnraw:
    return data.view(numpy.recarray)
  if magnitude:
    func = RectBivariateSpline(zbins, - data['M_AB'], data['Phi'].T)
    func.x = zbins
    func.y = -data['M_AB']
    func.z = data['Phi'].T
    return func
  else:
    func = RectBivariateSpline(zbins, data['Lband'], data['Phi'].T)
    func.x = zbins
    func.y = data['Lband']
    func.z = data['Phi'].T
    return func
  
def Miz2(U, Lbol):
    """Shen etal 2009 """
    return 90.0 - 2.5 * numpy.log10(numpy.float64(Lbol) * 3.9e33)

def M(U, Lbol, band=bands.i, z=2.0):
    """ Richards etal 2006  arxiv 0601434v2 22 Feb 2006, EQ (4)
        i2 is the restframe of i band at reshift 2, or 2500 A rest frame.
        bhLbol gives the band luminosity at i2. We use cgs solar luminosity.

        Notice that Hopkins SED for 3C 273 gives M_i(z=2) of -28.6,
        assuming mdot = 4.0 (Lbol = 2.2e47 erg/s) different
        from number (-27.2) quoted from Richards. 

        Reducing mdot to 1.1 gives -27.2, meaning Lbol ~ 6.4e46; or Hopkins SED
        doesn't fit 3C273 well?
    """
    raise Exception("This is wrong do not use this, use Miz2 for sdss z=2 iband mag")
    A = _bandconv(U, band)
    A /= (1 + z)
    A /= U.METER
    L = Lband(U, Lbol, ('A', A * 1e10))
    freq = 3e8 / A
    return numpy.log10(L * 3.9e33 / freq\
            / (4 * 3.14 * 3.08e19**2)) / -0.4 - 48.6 - 2.5 * numpy.log10(1+z)

def Lband(U, Lbol, band=None):
  if band is None:
      return Lbol 
  if isinstance(band, basestring):
    params = {
      'bol': 0,
      'blue': -1,
      'ir':   -2, 
      'soft': -3,
      'hard': -4
    }
    nu = params[band]
  else:
    A = _bandconv(U, band)
    nu = U.C / A / U.HERTZ
  return  _qlf.Lband(Lbol, nu)

def bhLbol(U, mdot, band=None):
  """ converts accretion rate in internal units to bolemetric, 
  in units of Lsun,
      
  """
  Lbol = mdot * U.C ** 2 * 0.1 / U.SOLARLUMINOSITY
  return Lband(U, Lbol, band)

def bolemetric(U, Lbol, band, photon=False):
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
        if photon is True and band is not bol blue ir or soft hard,
        returns in units of photons / sec.
    """ 
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
      Arange = _bandconv(U, band)
      if numpy.isscalar(Arange):
        Amin = _bandconv(U, band)
        Amax = Amin
      else:
        Amin, Amax = _bandconv(U, band)

      if Amin < 49e-10 * U.METER \
      or Amax > 1e-6 * U.METER:
        raise ValueError('input band %s(%g A, %g A) not implemented' % (band, Amin / U.METER * 1e10, Amax / U.METER * 1e10))
    else:
      c1,k1,c2,k2 = params[band]

    x = numpy.float64(Lbol * 1e-10)
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
        # this is awkward. but the integral is on mu
        # yet the formula is written in terms of A(wavelength)
        # but it is correct.
        alpha = numpy.asarray(alpha)
        if (Amin > Ahigh) or (Amax < Alow): return 0
        Amin_, Amax_ = numpy.clip([Amin, Amax], Alow, Ahigh)
        lmax = lref * (Aref / Amax_) ** alpha
        if Amin == Amax:
            # return specific luminosity * mu
            return lmax / Amax * U.C
        if not photon:
          return lmax / Amax_ * U.C / (alpha + 1) * \
                ((Amax_ / Amin_) ** (alpha + 1) - 1)
        else:
          return lmax * U.SOLARLUMINOSITY / U.PLANCK * U.SECOND / alpha * \
                  ((Amax_ / Amin_)  ** alpha - 1)

      res = [
            # this piece bridges UV and X ray
            # we assume -1.76 but it seems to be between -1.76 and alphaOX,
            # very complicated stuff. GABE's old code seems to be assuming -1.76, too.
             piece(A50, A500, alpha50, l50, A50), 
            # UV piece
             piece(A500, A1200, -1.76, l1200, A1200),
            # Optical piece
             piece(A1200, A4450 * 10, -0.44, l4450, A4450)]

      #print Amin, Amax, res
      return numpy.where(Lbol == 0, 0, reduce(numpy.add,res))
    else:
      Lband = Lbol / ratio
    return Lband
def bondi(U, mass, rho, cs, relvel, boost=100):
    """ returns the bondi rate """
    norm = (cs * cs + relvel * relvel) ** 1.5;
    mdot = 4. * 3.14 * boost * U.G ** 2 \
            * mass ** 2  * rho / norm
    return mdot

def eddington(U, mass, factor=3.):
  """ returns the eddington limited growth rate """
  return 4 * numpy.pi * U.G * U.C * U.PROTONMASS / (0.1 * U.C ** 2 *
      U.THOMSON_CROSSSECTION) * factor * mass

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
