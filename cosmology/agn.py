import numpy

def bolemetric(units, mdot, band):
    """ Hopkins etal 2007, added UV (13.6ev -> 250.0 ev with index=1.76),
        type can be 'bol', 'blue', 'ir', 'soft', 'hard', 'uv'
        or one of the follows:
        - (min, max), in A,
        - ('A', min, max), in A,
        - ('ev', min, max), in ev,
        'uv' ::= ('ev', 13.6, 250.0)
        total band lumisoty is returned, and only from 50A to 1um is covered
        returns the bolemtric luminosity in 'units' of power
    """
    params = {
      'blue': (6.25,-0.37,9.00,-0.012,),
      'ir':   (7.40,-0.37, 10.66,-0.014,),
      'soft':(17.87,0.28,10.03,-0.020,),
      'hard':(10.83,0.28,6.08,-0.020,),
    }

    if band == 'uv':
      band = ('ev', 13.6, 250.)

    A1200 = 1200e-10 * units.METER
    A500 = 500e-10 * units.METER
    A50 = 50e-10 * units.METER   # about 250 ev
    A2500 = 2500e-10 * units.METER
    A4450 = 4450e-10 * units.METER
    A2k = units.C * units.PLANCK / (2000 * units.EV) # about 6.2 A

    L = mdot * units.C ** 2 * 0.1
    if band == 'bol': return L 

    if band not in params:
      c1,k1,c2,k2 = params['blue']
      # extrapolate from lue band luminosity
      if len(band) == 3:
        if band[0] == 'A':
          Amin = band[1] * units.METER * 1e-10
          Amax = band[2] * units.METER * 1e-10
        elif band[0] == 'ev':
          Amin = units.PLANCK * units.C / (units.EV * band[2])
          Amax = units.PLANCK * units.C / (units.EV * band[1])
        else:
          raise ValueError('format of band is (min, max), ("A", min, max), or ("ev", min, max)')
      elif len(band) == 2: 
        Amin = band[0] * units.METER * 1e-10
        Amax = band[1] * units.METER * 1e-10
      else:
        raise ValueError('format of band is (min, max), ("A", min, max), or ("ev", min, max)')
      if Amin < 49e-10 * units.METER \
      or Amax > 1e-6 * units.METER:
        raise ValueError('input band %s(%g A, %g A) not implemented' % (band, Amin / units.METER * 1e10, Amax / units.METER * 1e10))
    else:
      c1,k1,c2,k2 = params[band]

    x = L / (1e10 * units.SOLARLUMINOSITY)
    ratio = c1 * x ** k1 + c2 * x ** k2

    if band not in params:
      L4450 = L / ratio
      l4450 = L4450 / (units.C / A4450)
      l1200 = l4450 * (A4450 / A1200) ** -0.44
      l500 = l1200 * (A1200 / A500) ** -1.76
      l2500 = l4450 * (A4450 / A2500) ** -0.44
      alphaOX = -0.107 * numpy.log10(l2500 / units.ERG) + 1.739
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
        return lmax / Amax_ * units.C / (alpha + 1) * ((Amax_ / Amin_) ** (alpha + 1) - 1)

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
      return numpy.sum(res)
    else:
      Lband = L / ratio
    return Lband

def QSObol(units, mdot, type):
    """ this code is incorrectly handling 50-500A.
       Hopkins etal 2007, added UV (13.6ev -> 250.0 ev with index=1.76),
type can be 'blue', 'ir', 'soft', 'hard', or 'uv'. """
    params = {
      'blue': (6.25,-0.37,9.00,-0.012,),
      'ir': (7.40,-0.37, 10.66,-0.014,),
      'soft':(17.87,0.28,10.03,-0.020,),
      'hard':(10.83,0.28,6.08,-0.020,),
    }

    L = mdot * units.C ** 2 * 0.1
    if type == 'bol': return L

    if type == 'uv':
      c1,k1,c2,k2 = params['blue']
    else:
      c1,k1,c2,k2 = params[type]
    x = L / (1e10 * units.SOLARLUMINOSITY)
    ratio = c1 * x ** k1 + c2 * x ** k2
    if type == 'uv':
      LB = L / ratio
      fB = units.C / (445e-9 * units.METER)
      fX = units.C / (120e-9 * units.METER)
      fI = units.C / (91.1e-9 * units.METER)
      lB = LB / fB
      lX = lB * (fX / fB) ** -0.44
      lI = lX * (fI / fX) ** -1.76
      print 'lI = ', lI
      Lband = lI * fI / -0.76 * ((250.0 / 13.6) ** -0.76 - 1)
    else:
      Lband = L / ratio
    return Lband
if False:
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
