from numpy import linspace
from scipy.integrate import quad
from numpy import max
from numpy import array
from numpy import sqrt
import gadget.ccode
k0 = gadget.ccode.kernel.k0
kline = gadget.ccode.kernel.kline
koverlap = gadget.ccode.kernel.koverlap
akline = gadget.ccode.kernel.akline
akoverlap = gadget.ccode.kernel.akoverlap

def kernel(eta) :
  """ dimensionless sph kernel
      eta = r / smoothinglength
      to get the real kernel, divide the return value by smoothinglength**3
  """
  return k0(eta)

def init_akline() :
  bins = akline.size
  etas = linspace(0, 1, bins)
  deta = 1.0/bins
  akline[:] = 2.0 * array([
              quad(lambda x,eta : k0(sqrt(x**2+eta**2)),
                   0, sqrt(1.0 - eta**2), args=(eta))[0]
              for eta in etas
              ])[:]
  return 

def init_akoverlap() :
  gadget.ccode.kernel.fill_koverlap()
  akoverlap[:,:,:,:] /= akoverlap.max()

init_akline()
init_akoverlap()
