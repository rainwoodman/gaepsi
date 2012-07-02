#cython: embedsignature=True
#cython: cdivision=True
import numpy
cimport cpython
cimport numpy
from libc.stdint cimport *

cdef extern from 'math.h':
  cdef int fmax(double, double) nogil

cdef void decode(int64_t key, int32_t point[3]) nogil
cdef int64_t encode(int32_t point[3]) nogil
cdef int boxtest (int64_t ind, int order, int64_t key) nogil 
cdef int AABBtest(int64_t ind, int order, int64_t AABB[2]) nogil 
cdef void diff(int64_t p1, int64_t p2, int32_t d[3]) nogil

cdef class Digitize:
  """Zorder scales x,y,z to 0 ~ (1<<bits) - 1 """
  cdef double * _min
  cdef double _norm
  cdef double _Inorm
  cdef readonly int bits
  cdef readonly numpy.ndarray min
  cdef readonly numpy.ndarray scale

  cdef inline void i2f(self, int32_t point[3], double pos[3]) nogil:
    cdef int d
    for d in range(3):
      pos[d] = point[d] * self._Inorm + self._min[d]
   
  cdef inline void f2i(self, double pos[3], int32_t point[3]) nogil:
    """ this will round to 0 , (1<<21)  -1"""
    cdef int d
    for d in range(3):
      point[d] = <int32_t> fmax(0, (pos[d] - self._min[d]) * self._norm)
      if point[d] >= (1<<21): point[d] = 0x1fffff
