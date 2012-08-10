
#cython: embedsignature=True
#cython: cdivision=True
import numpy
cimport cpython
cimport numpy
cimport npyiter
from libc.stdint cimport *
from libc.math cimport cos, sin, sqrt, fabs, acos, atan2
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

    cdef npyiter.CIter citer
    cdef size_t size = npyiter.init(&citer, iter)
    cdef double RA1, RA2, DEC1, DEC2, ANG
    with nogil: 
      while size > 0:
        while size > 0:
          RA1 = (<double*>citer.data[0])[0]
          DEC1 = (<double*>citer.data[1])[0]
          RA2 = (<double*>citer.data[2])[0]
          DEC2 = (<double*>citer.data[3])[0]
          # z = sin(dec), this is correct
          ANG = cos(RA1 - RA2) * cos(DEC1) * cos(DEC2) + sin(DEC1) * sin(DEC2)
          
          if ANG > 1.0: ANG = 1.0
          if ANG < -1.0: ANG = -1.0
          (<double*>citer.data[4])[0] = acos(ANG)
          npyiter.advance(&citer)
          size = size - 1
        size = npyiter.next(&citer)
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

    cdef npyiter.CIter citer
    cdef size_t size = npyiter.init(&citer, iter)
    cdef double a, b, T, c
    with nogil: 
      while size > 0:
        while size > 0:
          a = (<double*>citer.data[0])[0]
          b = (<double*>citer.data[1])[0]
          T = (<double*>citer.data[2])[0]
          
          c = sqrt(fabs(a * a + b * b - 2 * a * b * cos(T)))
          (<double*>citer.data[3])[0] = c
          npyiter.advance(&citer)
          size = size - 1
        size = npyiter.next(&citer)
    return out

def radec2pos(ra, dec, Dc, out=None):
    """ only for flat comoving coordinate is returned as (-1, 3) 
    """
    shape = numpy.broadcast(dec, ra, Dc).shape
    if out is None:
      out = numpy.empty(shape=shape, dtype=('f4', 3))

    iter = numpy.nditer(
          [ra, dec, Dc, out[..., 0], out[..., 1], out[..., 2]], 
      op_flags=[['readonly'], ['readonly'], ['readonly'], 
               ['writeonly'], ['writeonly'], ['writeonly']], 
     op_dtypes=['f8', 'f8', 'f8', 'f8', 'f8', 'f8'],
         flags=['buffered', 'external_loop'], 
       casting='unsafe')

    cdef npyiter.CIter citer
    cdef size_t size = npyiter.init(&citer, iter)
    cdef double RA, DEC, DC, OUT0, OUT1, OUT2
    with nogil: 
      while size > 0:
        while size > 0:
          RA = (<double*>citer.data[0])[0]
          DEC = (<double*>citer.data[1])[0]
          DC = (<double*>citer.data[2])[0]
          
          OUT0 = DC * cos(DEC) * cos(RA)
          OUT1 = DC * cos(DEC) * sin(RA)
          OUT2 = DC * sin(DEC)

          (<double*>citer.data[3])[0] = OUT0
          (<double*>citer.data[4])[0] = OUT1
          (<double*>citer.data[5])[0] = OUT2

          npyiter.advance(&citer)
          size = size - 1
        size = npyiter.next(&citer)
    return out

def pos2radec(pos, out=None):
    """ only for flat comoving coordinate is input 
        returns ra, dec, Dc.
    """
    if out is None:
      out = numpy.empty(shape=pos.shape, dtype=('f4'))

    
    iter = numpy.nditer(
          [pos[..., 0], pos[..., 1], pos[..., 2], out[..., 0], out[..., 1], out[..., 2]], 
      op_flags=[['readonly'], ['readonly'], ['readonly'], 
               ['writeonly'], ['writeonly'], ['writeonly']], 
     op_dtypes=['f8', 'f8', 'f8', 'f8', 'f8', 'f8'],
         flags=['buffered', 'external_loop'], 
       casting='unsafe')

    cdef npyiter.CIter citer
    cdef size_t size = npyiter.init(&citer, iter)
    cdef double RA, DEC, DC, X, Y, Z
    with nogil: 
      while size > 0:
        while size > 0:
          X = (<double*>citer.data[0])[0]
          Y = (<double*>citer.data[1])[0]
          Z = (<double*>citer.data[2])[0]

          DC = sqrt(X * X + Y * Y + Z * Z)
          RA = atan2(Y, X)
          DEC = atan2(Z, DC)

          (<double*>citer.data[3])[0] = RA
          (<double*>citer.data[4])[0] = DEC
          (<double*>citer.data[5])[0] = DC

          npyiter.advance(&citer)
          size = size - 1
        size = npyiter.next(&citer)
    return out

ctypedef double (*method_t) (Cosmology, double) nogil
cdef class Cosmology:
  """ this is dimensionless
      H is in unit of H0,
      distance is in unit of DH or C/H0
      time is in unit of 1/H0
  """
  cdef double M
  cdef double K
  cdef double L
  cdef double h
  cdef double R
  methods = {
      'eta': <intptr_t>Cosmology.eta_one,
      'ddplus': <intptr_t>Cosmology.ddplus_one,
      'dladt': <intptr_t>Cosmology.dladt_one,
      'H': <intptr_t>Cosmology.H_one,
    }
  def __cinit__(self, double M, double K, double L, double h):
    self.M = M
    self.K = K
    self.L = L
    self.h = h
    self.R = 0

  cdef double H_one(self, double a) nogil:
    cdef double a1 = 1.0 / a
    cdef double a2 = a1 * a1
    cdef double a3 = a1 * a2
    cdef double a4 = a2 * a2
    return sqrt(self.M * a3 + self.R * a4 + self.K * a2 + self.L)

  cdef double eta_one(self, double a) nogil:
    return a * self.H_one(a)
  cdef double ddplus_one(self, double a) nogil:
    return 2.5 * self.eta_one(a) ** -3
  cdef double dladt_one(self, double a) nogil:
    return a * self.eta_one(a)

  def eval(self, func, input, output=None):
    if output is None:
      output = numpy.empty_like(input, 'f8')

    iter = numpy.nditer([input, output], 
      op_dtypes=['f8', 'f8'],
      op_flags=[['readonly'], ['writeonly']],
      flags=['external_loop', 'buffered', 'zerosize_ok'],
      casting='unsafe')

    cdef method_t method = <method_t><intptr_t>self.methods[func]

    cdef npyiter.CIter citer
    cdef size_t size = npyiter.init(&citer, iter)
    with nogil:
      while size > 0:
        while size > 0:
          (<double*>citer.data[1])[0] = method(self, (<double*>citer.data[0])[0])
          npyiter.advance(&citer)
          size = size - 1
        size = npyiter.next(&citer)
    return output

