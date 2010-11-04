import scipy.weave
from numpy import zeros
from numpy import linspace
from scipy.integrate import quad
from numpy import max
from numpy import meshgrid
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

  bins = akoverlap.shape[0]
  etaxs = linspace(-1, 1, bins + 1)
  etays = linspace(-1, 1, bins + 1)
  
  deta = 2.0 / bins
  etaxs = etaxs[:-1] + deta * 0.5
  etays = etays[:-1] + deta * 0.5

  X, Y = meshgrid(etaxs, etays)
  kl = kline(sqrt(X ** 2 + Y ** 2))

  a = zeros((bins, bins, bins, bins), dtype='f4')
  rowsum = zeros((bins, bins), dtype='f4')
  ccode = """
   for(int i0 = 0; i0 < bins; i0++)
   for(int j0 = 0; j0 < bins; j0++) {
     for(int i1 = i0; i1 < bins; i1++) {
       ROWSUM2(i1, j0) = KL2(i1, j0);
       for(int j1 = j0 + 1; j1 < bins; j1++)
         ROWSUM2(i1,j1) = ROWSUM2(i1,j1-1) + KL2(i1,j1);
     }

     for(int j1 = j0; j1 < bins; j1++) {
       A4(i0,j0,i0,j1) = ROWSUM2(i0,j1);
       for(int i1 = i0 + 1; i1 < bins; i1++) {
         A4(i0,j0,i1,j1) = A4(i0,j0,i1 - 1,j1) + ROWSUM2(i1,j1);
       }
     }
   }
  """
  scipy.weave.inline(ccode, ['bins', 'a', 'kl', 'rowsum'], extra_compile_args = ['-Wno-unused']);
  norm = max(a[0,0,:,:])
  print "normalization = ", norm * deta ** 2
  a /= norm
  akoverlap[:,:,:,:]= a[:,:,:,:]
  return a

init_akline()
init_akoverlap()
