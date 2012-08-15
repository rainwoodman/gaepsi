cimport numpy
import numpy
cimport npyiter
from libc.stdint cimport *
cdef extern from "qlf_calculator.c":
  int work (double nu, double z, int FIT_KEY, int npts, double * l_bol_grid, double * l_band_grid, double * M_AB_grid, double * S_nu_grid, double * phi_bol_grid) nogil
  double l_band_jacobian(double log_l_bol, double nu) nogil
  double l_band(double log_l_bol, double nu) nogil
  double l_band_dispersion(double log_l_bol, double nu) nogil

numpy.import_array()
ctypedef double (*method_t) (double, double) nogil

methods = {
  'jacobian': <intptr_t> l_band_jacobian,
  'dispersion': <intptr_t> l_band_dispersion,
  'l': <intptr_t> l_band,
}
def eval(method, Lbol, double nu, out=None):
  if out is None:
    out = numpy.empty_like(Lbol, dtype='f8')

  iter = numpy.nditer([Lbol, out],
       op_dtypes = ['f8', 'f8'],
       op_flags = [['readonly'], ['writeonly']],
       flags = ['external_loop', 'zerosize_ok', 'buffered'],
       casting = ['unsafe'])
  cdef npyiter.CIter citer
  cdef size_t size = npyiter.init(&citer, iter)
  cdef method_t func = methods[method]
  with nogil:
    while size > 0:
      while size > 0:
        (<double *>citer.data[1])[0] = func((<double *>citer.data[0])[0], nu)
        npyiter.advance(&citer)
        size = size - 1
      size = npyiter.next(&citer)
  return out

def qlf(double nu, double z, numpy.ndarray [double, ndim=1] Lbol, int fit_key=0):
  """input Lbol is in unit of Lsun
    nu:
        nu = 0: bolometric
        nu = -1: B-band
        nu = -2: mid_IR
        nu = -3: soft X-ray(.5-2kev)
        nu = -4: hard X-ray(2-10kev)
        otherwise nu in Hz.
   fit_key -- optional flag which sets the fitted form of the bolometric QLF to 
        adopt in calculating the observed QLF at nu. The fit parameters and 
        descriptions are given in HRH06, 
        specifically Tables 3 & 4

        0 = "Full" (default) the best-fit model derived therein 
        (double power law fit, allowing both  
        the bright and faint-end slopes to evolve with redshift)
        chi^2/nu = 1007/508  (with respect to the compiled observations above)
        1 = "PLE" 
        (pure luminosity evolution - double power-law with evolving Lstar but 
        no change in the faint & bright end slopes)
        chi^2/nu = 1924/511 
        2 = "Faint" 
        (double power-law with evolving Lstar and evolving faint-end slope,  
        but fixed bright end slope)
        chi^2/nu = 1422/510 
        3 = "Bright" 
        (double power-law with evolving Lstar and evolving bright-end slope,  
        but fixed faint end slope)
        chi^2/nu = 1312/509 
        4 = "Scatter" 
        (double power-law with evolution in both faint and bright end slopes, 
        but adding a uniform ~0.05 dex to the error estimates sample-to-sample, 
        to *roughly* compare systematic offsets)
        chi^2/nu = 445/508 (note, this fit has the error bars increased, and 
        also under-weights some of the most well-constrained samples, 
        so the reduced chi^2/nu should not be taken to mean it is 
        necessarily more accurate)
        5 = "Schechter"
        (fitting a modified Schechter function, allowing both bright and 
        faint end slopes to evolve with redshift, instead of a double power law)
        chi^2/nu = 1254/509 
        6 = "LDDE"
        (luminosity-dependent density evolution fit)
        chi^2/nu = 1389/507
        7 = "PDE"
        (pure density evolution fit)
        chi^2/nu = 3255/511

     returns:
     Lband: the band luminosity in given by nu in Lsun,
     M_AB: the monochromatic AB absolution magnitude
     S_nu: the ovserved specific flux WITHOUT K-correction or bandpass redshifting in MJy
     Phi: the number density per unit log_{10}(L in Mpc^{-3}) for the band
  """
  cdef numpy.ndarray Lband = numpy.empty_like(Lbol, dtype='f8')
  cdef numpy.ndarray M_AB = numpy.empty_like(Lbol, dtype='f8')
  cdef numpy.ndarray S_nu = numpy.empty_like(Lbol, dtype='f8')
  cdef numpy.ndarray Phi = numpy.empty_like(Lbol, dtype='f8')
  cdef size_t size = Lbol.size
  with nogil:
    work(nu, z, fit_key, size, <double*>Lbol.data, <double*>Lband.data, <double*>M_AB.data, <double*>S_nu.data, <double*>Phi.data) 
  return Lband, M_AB, S_nu, Phi
