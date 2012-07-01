#cython: embedsignature=True
#cython: cdivision=True
import numpy
cimport cpython
from cpython.ref cimport Py_INCREF
cimport numpy
cimport npyiter
from libc.stdint cimport *
from libc.stdlib cimport malloc, realloc, free
from libc.float cimport FLT_MAX
from libc.limits cimport INT_MAX, INT_MIN
from libc.math cimport fmin
cimport cython
import cython
from warnings import warn

numpy.import_array()

cdef class Zorder:
  """Scale scales x,y,z to 0 ~ (1<<bits) - 1 """
  def __cinit__(self):
    self.min = numpy.empty(3)
    self._min = <double*>self.min.data
    self.scale = numpy.empty(3)

  @classmethod
  def from_points(klass, x, y, z, bits=21):
    return klass(
      min=numpy.array([x.min(), y.min(), z.min()]),
      scale=numpy.array([ x.ptp(), y.ptp(), z.ptp()]),
      bits = bits
    )

  def __init__(self, min, scale, bits=21):
    if bits > 21:
      raise ValueError("bits cannnot be bigger than 21 with 64bit integer")
    self.min[:] = min
    self.scale[:] = scale
    self.bits = bits
    self._norm = 1.0 / self.scale.max() * ((1 << bits) -1)
    self._Inorm = 1.0 / self._norm

  cdef void decode(Zorder self, int64_t key, int32_t point[3]) nogil:
    cdef int j
    ind2xyz(key, point, point+1, point+2)
    return
  cdef int64_t encode(Zorder self, int32_t point[3]) nogil:
    return xyz2ind(point[0], point[1], point[2])

  def invert(self, index, out=None):
    """ revert from zorder indices to floating points """
    if out is None:
      out = numpy.empty(numpy.broadcast(index).shape, dtype=('f4', 3))
    x, y, z = out[:, 0], out[:, 1], out[:, 2]
    iter = numpy.nditer([x, y, z, out], 
          op_flags=[['writeonly'], ['writeonly'], ['writeonly'], ['readonly']], 
          flags=['buffered', 'external_loop'], 
          casting='unsafe', 
          op_dtypes=['f4', 'f4', 'f4', 'i8'])

    cdef npyiter.CIter citer
    cdef size_t size = npyiter.init(&citer, iter)
    cdef double pos[3]
    with nogil:
      while size > 0:
        while size > 0:
          self.decode_float((<int64_t*>citer.data[3])[0], pos)
          (<float*>citer.data[0])[0] = pos[0]
          (<float*>citer.data[1])[0] = pos[1]
          (<float*>citer.data[2])[0] = pos[2]
          npyiter.advance(&citer)
          size = size - 1
        size = npyiter.next(&citer)
    
  def __call__(self, x, y, z, out=None):
    """ calculates the zorder of given points """

    if out is None:
      out = numpy.empty(numpy.broadcast(x, y, z).shape, dtype='i8')

    iter = numpy.nditer([x, y, z, out], 
          op_flags=[['readonly'], ['readonly'], ['readonly'], ['writeonly']], 
          flags=['buffered', 'external_loop'], 
          casting='unsafe', 
          op_dtypes=['f8', 'f8', 'f8', 'i8'])

    cdef npyiter.CIter citer
    cdef size_t size = npyiter.init(&citer, iter)

    cdef double pos[3]
    with nogil:
      while size > 0:
        while size > 0:
          for d in range(3):
            pos[d] = (<double*>citer.data[d])[0]

          (<int64_t*>citer.data[3])[0] = self.encode_float(pos)

          npyiter.advance(&citer)
          size = size - 1
        size = npyiter.next(&citer)

  def __str__(self):
    return str(dict(min=self.min, scale=self.scale, bits=self.bits))
  def __repr__(self):
    return str(dict(min=self.min, scale=self.scale, bits=self.bits))

  cdef void BBint(self, double pos[3], float r, int32_t center[3], int32_t min[3], int32_t max[3]) nogil:
    cdef double rf
    for d in range(3):
      center[d] = <int32_t> ((pos[d] - self._min[d] ) * self._norm)
      rf = r * self._norm
      min[d] = <int32_t>(<double>center[d] - rf)
      max[d] = <int32_t>(<double>center[d] + rf)
