# cython:cdivision=True

cdef extern from """math.h"
typedef __int128_t fckey_t;
typedef __uint128_t ufckey_t;
typedef int64_t ipos_t;
#define ______ \"""":
  ctypedef int fckey_t
  ctypedef unsigned int ufckey_t
  ctypedef int ipos_t

cdef int f2i(double scale[4], double pos[3], ipos_t point[3]) nogil
cdef void i2f(double scale[4], ipos_t point[3], double pos[3]) nogil
cdef void i2fc(ipos_t ipos[3], fckey_t * key) nogil
cdef void fc2i(fckey_t * key, ipos_t ipos[3]) nogil
