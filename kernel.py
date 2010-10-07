import scipy.weave
from numpy import isscalar
from numpy import zeros
from numpy import linspace
from scipy.integrate import quad
from numpy import max
from numpy import meshgrid
from numpy import array
from numpy import sqrt
from numpy import pi
from numpy import zeros_like
from numpy import int32


def kernel(eta) :
  """ dimensionless sph kernel
      eta = r / smoothinglength
      to get the real kernel, divide the return value by smoothinglength**3
  """
  if isscalar(eta):
    if eta < 0.5:
      return 8 / pi * (1.0 - 6 * eta **2 + 6 * eta **3)
    if eta < 1.0:
      return 8 / pi * 2.0 * (1.0 - eta)**3
    return 0
  else :
    mask1 = eta < 0.5
    mask2 = (eta < 1.0) & (eta >= 0.5)
    result = zeros_like(eta)
    result[mask1] = 8 / pi * (1.0 - 6 * eta[mask1] **2 + 6 * eta[mask1] ** 3)
    result[mask2] = 8 / pi * 2.0 * (1.0 - eta[mask2]) ** 3
    return result

kernel_line_values = None
kernel_line_bins = None
kernel_line_deta = None
def init_kernel_line(bins=1000) :
  global kernel_line_values
  global kernel_line_deta
  global kernel_line_bins
  kernel_line_bins = bins
  etas = linspace(0, 1, bins)
  kernel_line_deta = 1.0/bins
  kernel_line_values = 2.0 * array([
              quad(lambda x,eta : kernel(sqrt(x**2+eta**2)),
                   0, sqrt(1.0 - eta**2), args=(eta))[0]
              for eta in etas
              ])
  return etas, kernel_line_values

kernel_box_values = None
kernel_box_deta = None

def init_kernel_box(bins=100) :
  global kernel_box_values
  global kernel_box_bins
  global kernel_box_deta

  kernel_box_bins = bins
  etaxs = linspace(-1, 1, bins + 1)
  etays = linspace(-1, 1, bins + 1)
  
  kernel_box_deta = 2.0 / bins
  etaxs = etaxs[:-1] + kernel_box_deta * 0.5
  etays = etays[:-1] + kernel_box_deta * 0.5

  X, Y = meshgrid(etaxs, etays)
  kl = kernel_line(sqrt(X ** 2 + Y ** 2))
  kernel_box_values = [ zeros_like(X), zeros_like(X), zeros_like(X), zeros_like(X)]

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
  scipy.weave.inline(ccode, ['bins', 'a', 'kl', 'rowsum'], extra_compile_args = ['-Wno-unused'], force = True);
  norm = max(a[0,0,:,:])
  print "normalization = ", norm * kernel_box_deta ** 2
  a /= norm
  kernel_box_values = a
  return a

def kernel_line(eta) :
  global kernel_line_values
  global kernel_line_bins
  global kernel_line_deta
  if kernel_line_values == None:
    init_kernel_line()
  if isscalar(eta):
    if eta < 1.0 :
      return kernel_line_values[int(eta / kernel_line_deta)]
    else :
      return 0.0
  else :
    mask = eta >= 1.0
    indices = zeros_like(eta)
    indices = int32(eta / kernel_line_deta)
    indices[mask] = 0
    ret = kernel_line_values[indices]
    ret[mask] = 0
    return ret

def kernel_box(etax0, etay0, etax1, etay1) :
  global kernel_box_values
  global kernel_box_deta
  global kernel_box_bins
  if kernel_box_values == None: init_kernel_box()
  boundary = array([ int32((1.0 + eta ) / kernel_box_deta) for eta in [etax0, etay0, etax1, etay1]])

  boundary[boundary >= kernel_box_bins] = kernel_box_bins -1
  boundary[boundary < 0] = 0
  i0,j0,i1,j1 = boundary
  return kernel_box_values[i0,j0,i1,j1]

init_kernel_box(100)
