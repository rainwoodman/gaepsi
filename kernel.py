from numpy import linspace
from scipy.integrate import quad
from numpy import max
from numpy import array
from numpy import sqrt

import ccode
k0 = ccode.k0
kline = ccode.kline
koverlap = ccode.koverlap
akline = ccode.akline
akoverlap = ccode.akoverlap
def kernel(eta) :
  """ dimensionless sph kernel
      eta = r / smoothinglength
      to get the real kernel, divide the return value by smoothinglength**3
  """
  return k0(eta)

def mk_akline() :
  bins = akline.size
  etas = linspace(0, 1, bins)
  deta = 1.0/bins
  return 2.0 * array([
              quad(lambda x,eta : k0(sqrt(x**2+eta**2)),
                   0, sqrt(1.0 - eta**2), args=(eta))[0]
              for eta in etas
              ])
def mk_aklinesq() :
  bins = akline.size
  etas = linspace(0, 1, bins)
  deta = 1.0/bins
  return 2.0 * array([
              quad(lambda x,eta : k0(sqrt(x**2+eta)),
                   0, sqrt(1.0 - eta), args=(eta))[0]
              for eta in etas
              ])

# these functions are no longer needed as the arrays are inited in the C module
#init_akline()
#init_akoverlap()
