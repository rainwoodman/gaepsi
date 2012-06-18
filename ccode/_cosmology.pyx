
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
from libc.math cimport M_1_PI, cos, sin, sqrt, fabs, acos
from libc.limits cimport INT_MAX, INT_MIN
from warnings import warn
cimport cython
import cython

numpy.import_array()

def sphdist(ra1, dec1, ra2, dec2, out=None):
    if out is None:
      out = numpy.empty(shape=numpy.broadcast(ra1, dec1, ra2, dec2).shape, dtype='f4')

    iter = numpy.nditer(
          [ra1, dec1, ra2, dec2, out], 
      op_flags=[['readonly'], ['readonly'], ['readonly'], 
                ['readonly'], ['writeonly']], 
     op_dtypes=['f8', 'f8', 'f8', 'f8', 'f8'],
         flags=['buffered', 'external_loop'], 
       casting='unsafe')

    cdef npyiter.NpyIter * citer = npyiter.GetNpyIter(iter)
    cdef npyiter.IterNextFunc next = npyiter.GetIterNext(citer, NULL)
    cdef char ** data = npyiter.GetDataPtrArray(citer)
    cdef numpy.npy_intp *strides = npyiter.GetInnerStrideArray(citer)
    cdef numpy.npy_intp *size_ptr = npyiter.GetInnerLoopSizePtr(citer)
    cdef intptr_t iop, size
    cdef double RA1, RA2, DEC1, DEC2, ANG
    with nogil: 
      while True:
        size = size_ptr[0]
        while size > 0:
          RA1 = (<double*>data[0])[0]
          DEC1 = (<double*>data[1])[0]
          RA2 = (<double*>data[2])[0]
          DEC2 = (<double*>data[3])[0]
          # z = sin(dec), this is correct
          ANG = cos(RA1 - RA2) * cos(DEC1) * cos(DEC2) + sin(DEC1) * sin(DEC2)
          
          if ANG > 1.0: ANG = 1.0
          if ANG < -1.0: ANG = -1.0
          (<double*>data[4])[0] = acos(ANG)
          for iop in range(5):
            data[iop] += strides[iop]
          size = size - 1
        if next(citer) == 0: break
    return out

def thirdleg(A, B, t, out=None):
    shape = numpy.broadcast(A, B, t).shape
    if out is None:
      out = numpy.empty(shape=shape, dtype='f4')
    iter = numpy.nditer(
          [A, B, t, out],
      op_flags=[['readonly'], ['readonly'], ['readonly'], 
               ['writeonly']], 
     op_dtypes=['f8', 'f8', 'f8', 'f8'],
         flags=['buffered', 'external_loop'], 
       casting='unsafe')

    cdef npyiter.NpyIter * citer = npyiter.GetNpyIter(iter)
    cdef npyiter.IterNextFunc next = npyiter.GetIterNext(citer, NULL)
    cdef char ** data = npyiter.GetDataPtrArray(citer)
    cdef numpy.npy_intp *strides = npyiter.GetInnerStrideArray(citer)
    cdef numpy.npy_intp *size_ptr = npyiter.GetInnerLoopSizePtr(citer)
    cdef intptr_t iop, size
    cdef double a, b, T, c
    with nogil: 
      while True:
        size = size_ptr[0]
        while size > 0:
          a = (<double*>data[0])[0]
          b = (<double*>data[1])[0]
          T = (<double*>data[2])[0]
          
          c = sqrt(fabs(a * a + b * b - 2 * a * b * cos(T)))
          (<double*>data[3])[0] = c

          for iop in range(4):
            data[iop] += strides[iop]

          size = size - 1
        if next(citer) == 0: break
    return out

def radec2pos(cosmology, ra, dec, z, out=None):
    """ only for flat cosmology, comoving coordinate is returned as (-1, 3) 
        ra cannot be an alias of out[:, 0], out[:, 2]
        dec cannot be an alias of out[:, 0]
        
    """
    shape = numpy.broadcast(dec, ra, z).shape
    if out is None:
      out = numpy.empty(shape=shape, dtype=('f4', 3))

    Dc = cosmology.Dc(z)

    iter = numpy.nditer(
          [ra, dec, Dc, out[..., 0], out[..., 1], out[..., 2]], 
      op_flags=[['readonly'], ['readonly'], ['readonly'], 
               ['writeonly'], ['writeonly'], ['writeonly']], 
     op_dtypes=['f8', 'f8', 'f8', 'f8', 'f8', 'f8'],
         flags=['buffered', 'external_loop'], 
       casting='unsafe')

    cdef npyiter.NpyIter * citer = npyiter.GetNpyIter(iter)
    cdef npyiter.IterNextFunc next = npyiter.GetIterNext(citer, NULL)
    cdef char ** data = npyiter.GetDataPtrArray(citer)
    cdef numpy.npy_intp *strides = npyiter.GetInnerStrideArray(citer)
    cdef numpy.npy_intp *size_ptr = npyiter.GetInnerLoopSizePtr(citer)
    cdef intptr_t iop, size
    cdef double RA, DEC, DC, OUT0, OUT1, OUT2
    with nogil: 
      while True:
        size = size_ptr[0]
        while size > 0:
          RA = (<double*>data[0])[0]
          DEC = (<double*>data[1])[0]
          DC = (<double*>data[2])[0]
          
          OUT0 = DC * cos(DEC) * cos(RA)
          OUT1 = DC * cos(DEC) * sin(RA)
          OUT2 = DC * sin(DEC)

          (<double*>data[3])[0] = OUT0
          (<double*>data[4])[0] = OUT1
          (<double*>data[5])[0] = OUT2

          for iop in range(6):
            data[iop] += strides[iop]

          size = size - 1
        if next(citer) == 0: break
    return out
