from gaepsi._gaepsiccode import pmin
from gaepsi._gaepsiccode import k0, kline, koverlap, akline, akoverlap
from gaepsi._gaepsiccode import sml
from gaepsi._gaepsiccode import peanohilbert
from gaepsi._gaepsiccode import scanline
from gaepsi._gaepsiccode import camera
from gaepsi._gaepsiccode import merge as _merge, permute as _permute
#from gaepsi.ccode import ztree as ztree
from render import color
from image import rasterize

import numpy
def merge(data, A, B, out=None):
  if out is None:
    out = numpy.empty(len(A) + len(B), dtype='i8')
  if A.dtype is not numpy.dtype('i8'):
    raise ValueError("only i8 is supported for A")
  if B.dtype is not numpy.dtype('i8'):
    raise ValueError("only u8 is supported for B")
  if out.dtype is not numpy.dtype('i8'):
    raise ValueError("only u8 is supported for out")

  _merge(data, A, B, out)
  return out
def permute(array, index):
  if len(array.shape) > 1: 
    raise ValueError("array has to be 1D")
  if index.dtype is not numpy.dtype('i8') and index.dtype is not numpy.dtype('u8'):
    raise ValueError("index has to be i8 or u8")
    
  return _permute(array, index)

