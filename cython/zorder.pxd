#cython: embedsignature=True
#cython: cdivision=True
import numpy
cimport cpython
cimport numpy
from libc.stdint cimport *
cdef extern from '_bittricks.c':
  cdef int64_t xyz2ind (int32_t x, int32_t y, int32_t z) nogil 
  cdef void ind2xyz (int64_t ind, int32_t* x, int32_t* y, int32_t* z) nogil

cdef class Zorder:
  """Zorder scales x,y,z to 0 ~ (1<<bits) - 1 """
  cdef double * _min
  cdef double _norm
  cdef double _Inorm
  cdef readonly int bits
  cdef readonly numpy.ndarray min
  cdef readonly numpy.ndarray scale

  cdef void decode(Zorder self, int64_t key, int32_t point[3]) nogil
  cdef int64_t encode(Zorder self, int32_t point[3]) nogil

  cdef inline float dist2(Zorder self, int32_t center[3], int32_t point[3]) nogil:
    """ returns the floating distance ** 2 of integer point from center """
    cdef float x, dx
    cdef int d
    x = 0
    for d in range(3):
       dx = (point[d] - center[d]) * self._Inorm
       x += dx * dx
    return x
    
  cdef inline void decode_float(Zorder self, int64_t key, float pos[3]) nogil:
    cdef int32_t point[3]
    cdef int d
    self.decode(key, point)
    for d in range(3):
      pos[d] = point[d] * self._Inorm + self._min[d]
  cdef inline int64_t encode_float (Zorder self, float pos[3]) nogil:
    cdef int32_t point[3]
    cdef int d
    for d in range(3):
      point[d] = <int32_t> ((pos[d] - self._min[d]) * self._norm)
    return self.encode(point)

  cdef inline void float_to_int(Zorder self, float pos[3], int32_t point[3]) nogil:
    cdef int d
    for d in range(3):
      point[d] = <int32_t> ((pos[d] - self._min[d]) * self._norm)

  cdef void BBint(Zorder self, float pos[3], float r, int32_t center[3], int32_t min[3], int32_t max[3]) nogil
