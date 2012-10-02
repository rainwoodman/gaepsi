# cython:cdivision=True

cdef extern from """math.h"
typedef __int128_t fckey_t;
typedef __uint128_t ufckey_t;
typedef int64_t ipos_t;
#define ______ \"""":
  ctypedef int fckey_t
  ctypedef unsigned int ufckey_t
  ctypedef int ipos_t

cdef bint keyinkey(fckey_t needle, fckey_t hey, int order) nogil
cdef int f2i(double scale[4], double pos[3], ipos_t point[3]) nogil
cdef void i2f(double scale[4], ipos_t point[3], double pos[3]) nogil
cdef void i2f0(double scale[4], ipos_t point[3], double pos[3]) nogil
cdef void i2fc(ipos_t ipos[3], fckey_t * key) nogil
cdef void fc2i(fckey_t key, ipos_t ipos[3]) nogil
cdef fckey_t truncate(fckey_t key, int order) nogil
cdef double key2key2(double scale[4], fckey_t key1, fckey_t key2) nogil
cdef int heyinAABB(fckey_t hey, int order, fckey_t AABB[2]) nogil
