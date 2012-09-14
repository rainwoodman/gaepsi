#cython: embedsignature=True
#cython: cdivision=True
import numpy
cimport cpython
cimport numpy
from libc.stdint cimport *

cdef extern from 'math.h':
  cdef int fmax(double, double) nogil
  ctypedef int __int128_t

# to change zorder_t also modify the typedef in _bittricks.c !
ctypedef __int128_t zorder_t
from fillingcurve cimport ipos_t

cdef numpy.dtype _zorder_dtype
cdef void decode(zorder_t key, ipos_t point[3]) nogil
cdef zorder_t truncate(zorder_t key, int order) nogil
cdef zorder_t encode(ipos_t point[3]) nogil
cdef int boxtest (zorder_t ind, int order, zorder_t key) nogil 
cdef int AABBtest(zorder_t ind, int order, zorder_t AABB[2]) nogil 
cdef void diff(zorder_t p1, zorder_t p2, ipos_t d[3]) nogil

cdef class Digitize:
  """Zorder scales x,y,z to 0 ~ (1<<bits) - 1 """
  cdef double * _min
  cdef readonly double _norm
  cdef readonly double _Inorm
  cdef readonly int bits
  cdef readonly numpy.ndarray min
  cdef readonly numpy.ndarray scale

  cdef inline void i2f0(self, ipos_t point[3], double pos[3]) nogil:
    cdef int d
    for d in range(3):
      pos[d] = point[d] * self._Inorm
    
  cdef inline void i2f(self, ipos_t point[3], double pos[3]) nogil:
    cdef int d
    for d in range(3):
      pos[d] = point[d] * self._Inorm + self._min[d]
   
  cdef inline int f2i(self, double pos[3], ipos_t point[3]) nogil:
    """ will return 0 and give junk if any of the dimension is not 0 , (1<<self.bits)  -1,
        otherwize return 1 and fill point[3]"""
    cdef int d
    cdef double f
    for d in range(3):
      f = (pos[d] - self._min[d]) * self._norm
      if f < 0 or f >= (<ipos_t>1 << self.bits) : return 0
      else: point[d] = <ipos_t> f
#      if point[d] >= (1 << self.bits): point[d] = (1 << self.bits) -1
    return 1
