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
    self.norm = numpy.empty(3)
    self._norm = <double*>self.norm.data
    self.Inorm = numpy.empty(3)
    self._Inorm = <double*>self.Inorm.data

  @classmethod
  def from_points(klass, x, y, z, bits=21):
    return klass(
      min=numpy.array([x.min(), y.min(), z.min()]),
      norm=numpy.array([ 1 / x.ptp(), 1 / y.ptp(), 1 / z.ptp()]) * ((1 << bits) -1),
      bits = bits
    )

  def __init__(self, min, norm, bits=21):
    if bits > 21:
      raise ValueError("bits cannnot be bigger than 21 with 64bit integer")
    self.min[:] = min
    self.norm[:] = norm
    self.bits = bits
    self.Inorm[:] = 1.0 / norm
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
    cdef npyiter.NpyIter * citer = npyiter.GetNpyIter(iter)
    cdef npyiter.IterNextFunc next = npyiter.GetIterNext(citer, NULL)
    cdef char ** data = npyiter.GetDataPtrArray(citer)
    cdef numpy.npy_intp *strides = npyiter.GetInnerStrideArray(citer)
    cdef numpy.npy_intp *size_ptr = npyiter.GetInnerLoopSizePtr(citer)
    cdef intptr_t iop, size
    cdef float pos[3]
    with nogil:
     while True:
      size = size_ptr[0]
      while size > 0:
        self.decode_float((<int64_t*>data[3])[0], pos)
        (<float*>data[0])[0] = pos[0]
        (<float*>data[1])[0] = pos[1]
        (<float*>data[2])[0] = pos[2]
        for iop in range(4):
          data[iop] += strides[iop]
        size = size - 1
      if next(citer) == 0: break
    
  def __call__(self, x, y, z, out=None):
    """ calculates the zorder of given points """

    if out is None:
      out = numpy.empty(numpy.broadcast(x, y, z).shape, dtype='i8')

    iter = numpy.nditer([x, y, z, out], 
          op_flags=[['readonly'], ['readonly'], ['readonly'], ['writeonly']], 
          flags=['buffered', 'external_loop'], 
          casting='unsafe', 
          op_dtypes=['f4', 'f4', 'f4', 'i8'])

    cdef npyiter.NpyIter * citer = npyiter.GetNpyIter(iter)
    cdef npyiter.IterNextFunc next = npyiter.GetIterNext(citer, NULL)
    cdef char ** data = npyiter.GetDataPtrArray(citer)
    cdef numpy.npy_intp *strides = npyiter.GetInnerStrideArray(citer)
    cdef numpy.npy_intp *size_ptr = npyiter.GetInnerLoopSizePtr(citer)
    cdef intptr_t iop, size
    cdef float pos[3]
    with nogil:
     while True:
      size = size_ptr[0]
      while size > 0:
        for d in range(3):
          pos[d] = (<float*>data[d])[0]

        (<int64_t*>data[3])[0] = self.encode_float(pos)

        for iop in range(4):
          data[iop] += strides[iop]
        size = size - 1
      if next(citer) == 0: break

  def __str__(self):
    return str(dict(min=self.min, norm=self.norm, bits=self.bits))

  cdef void BBint(self, float pos[3], float r, int32_t center[3], int32_t min[3], int32_t max[3]) nogil:
    cdef float rf, f
    for d in range(3):
      center[d] = <int32_t> ((pos[d] - self._min[d] ) * self._norm[d])
      rf = r * self._norm[d]
      f = center[d] - rf
      if f > INT_MAX: min[d] = INT_MAX
      elif f < INT_MIN: min[d] = INT_MIN
      else: min[d] = <int32_t>f

      f = center[d] + rf
      if f > INT_MAX: max[d] = INT_MAX
      elif f < INT_MIN: max[d] = INT_MIN
      else: max[d] = <int32_t>f
