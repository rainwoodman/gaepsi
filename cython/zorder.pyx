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

cdef extern from '_bittricks.c':
  cdef zorder_t _xyz2ind (int32_t x, int32_t y, int32_t z) nogil 
  cdef void _ind2xyz (zorder_t ind, int32_t* x, int32_t* y, int32_t* z) nogil
  cdef int _boxtest (zorder_t ind, int order, zorder_t key) nogil 
  cdef int _AABBtest(zorder_t ind, int order, zorder_t AABB[2]) nogil 
  cdef void _diff(zorder_t p1, zorder_t p2, int32_t d[3]) nogil

numpy.import_array()

cdef void decode(zorder_t key, int32_t point[3]) nogil:
    _ind2xyz(key, point, point+1, point+2)
cdef zorder_t encode(int32_t point[3]) nogil:
    return _xyz2ind(point[0], point[1], point[2])
cdef int boxtest (zorder_t ind, int order, zorder_t key) nogil:
  return _boxtest(ind, order, key)
cdef int AABBtest(zorder_t ind, int order, zorder_t AABB[2]) nogil:
  return _AABBtest(ind, order, AABB)
cdef void diff(zorder_t p1, zorder_t p2, int32_t d[3]) nogil:
  _diff(p1, p2, d)


cdef class Digitize:
  """Scale scales x,y,z to 0 ~ (1<<bits) - 1 """
  def __cinit__(self):
    self.min = numpy.empty(3)
    self._min = <double*>self.min.data
    self.scale = numpy.empty(3)

  @classmethod
  def adapt(klass, pos, bits=21):
    x, y, z = pos[..., 0], pos[..., 1], pos[..., 2]
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

  def invert(self, index, out=None):
    """ revert from zorder indices to floating points """
    if out is None:
      out = numpy.empty(numpy.broadcast(index, index).shape, dtype=('f8', 3))

    iter = numpy.nditer([out[..., 0], out[..., 1], out[..., 3], index], 
          op_flags=[['writeonly'], ['writeonly'], ['writeonly'], ['readonly']], 
          flags=['buffered', 'external_loop'], 
          casting='unsafe', 
          op_dtypes=['f8', 'f8', 'f8', 'i8'])

    cdef npyiter.CIter citer
    cdef size_t size = npyiter.init(&citer, iter)
    cdef int32_t ipos[3]
    cdef double fpos[3]
    cdef int d
    with nogil:
      while size > 0:
        while size > 0:
          decode((<zorder_t*>citer.data[3])[0], ipos)
          self.i2f(ipos, fpos)
          for d in range(3):
            (<double*>citer.data[d])[0] = fpos[d]
          npyiter.advance(&citer)
          size = size - 1
        size = npyiter.next(&citer)
    return out
  def __call__(self, pos, out=None):
    """ calculates the zorder of given points """

    if out is None:
      out = numpy.empty(numpy.broadcast(pos[..., 0], pos[..., 1]).shape, dtype='i8')

    iter = numpy.nditer([pos[..., 0], pos[..., 1], pos[..., 2], out], 
          op_flags=[['readonly'], ['readonly'], ['readonly'], ['writeonly']], 
          flags=['buffered', 'external_loop'], 
          casting='unsafe', 
          op_dtypes=['f8', 'f8', 'f8', 'i8'])

    cdef npyiter.CIter citer
    cdef size_t size = npyiter.init(&citer, iter)

    cdef int32_t ipos[3]
    cdef double fpos[3]
    cdef int d
    with nogil:
      while size > 0:
        while size > 0:
          for d in range(3):
            fpos[d] = (<double*>citer.data[d])[0]
          self.f2i(fpos, ipos)
          (<zorder_t*>citer.data[3])[0] = encode(ipos)
          npyiter.advance(&citer)
          size = size - 1
        size = npyiter.next(&citer)
    return out
  def __str__(self):
    return str(dict(min=self.min, scale=self.scale, bits=self.bits))
  def __repr__(self):
    return str(dict(min=self.min, scale=self.scale, bits=self.bits))
