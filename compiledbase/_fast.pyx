#cython: embedsignature=True
#cython: cdivision=True
cimport cpython
cimport numpy
cimport cython
import numpy
import cython
cimport npyufunc
from libc.stdint cimport *

from warnings import warn
cdef extern from 'math.h':
  double fmax(double, double) nogil
  double fmin(double, double) nogil
  float fmaxf(float, float) nogil
  float fminf(float, float) nogil
  int isinf(double) nogil

numpy.import_array()
numpy.import_ufunc()

from cython cimport floating

# Finite MAX
cdef inline float _finitemaxf(float a, float b) nogil:
  if isinf(a): return b
  if isinf(b): return a
  return fmaxf(a, b)

cdef inline double _finitemax(double a, double b) nogil:
  if isinf(a): return b
  if isinf(b): return a
  return fmax(a, b)

npyufunc.register(locals(), <void*>_finitemaxf, <void*>_finitemax, 2,
                   'reorderablenone',
                   "ffinitemax",
"""
   finitemax(a, b) 
   returns the larger of a b that is finite, ignore NaNs too.
"""
)

# Finite MIN
cdef inline float _finiteminf(float a, float b) nogil:
  if isinf(a): return b
  if isinf(b): return a
  return fminf(a, b)
cdef inline double _finitemin(double a, double b) nogil:
  if isinf(a): return b
  if isinf(b): return a
  return fmin(a, b)

npyufunc.register(locals(), <void*>_finiteminf, <void*>_finitemin, 2,
                   "reorderablenone",
                   "ffinitemin",
"""
   finitemin(a, b) 
   returns the larger of a b that is finite, ignore NaNs too.
"""
)

# wrap
cdef inline floating _wrap(floating a, floating b) nogil:
  while a >= b:
    a = a - b
  while a < 0:
    a += b
  return a

npyufunc.register(locals(), <void*>_wrap[float], <void*>_wrap[double], 2, 'reorderablenone',
                   "wrap",
  """
   wrap(x, p, out=None) 
   returns x wrapped within [0, p], it gets slow when x is far from [0, p)!
  """
)
__namespace = locals()
def finitemin(arr, axis=None):
  return __namespace['ffinitemin'].reduce(arr, axis=axis)
def finitemax(arr, axis=None):
  return __namespace['ffinitemax'].reduce(arr, axis=axis)
