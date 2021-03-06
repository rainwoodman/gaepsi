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
from libc.math cimport M_1_PI
from libc.limits cimport INT_MAX, INT_MIN
from warnings import warn
cimport cython
import cython

numpy.import_array()
cimport ztree

cdef float k0(float d, float h) nogil:
  if d == 0: return 8.0 * M_1_PI
  cdef float eta = d / h
  if eta < 0.5:
    return 8.0 * M_1_PI * (1.0 - 6 * (1.0 - eta) * eta * eta);
  if eta < 1.0:
    eta = 1.0 - eta;
    return 8.0 * M_1_PI * 2.0 * eta * eta * eta;
  return 0.0

cdef float addweight(float * r, float * w, float h, int NGB) nogil:
  cdef int i
  cdef float rt = 0
  for i in range(NGB):
    rt = rt + k0(r[i], h) * w[i]
  return rt

@cython.boundscheck(False)
cdef float solve_sml_one(float * r, float * w, int NGB, float mean) nogil:
  cdef float m
  cdef float hmin, hmax, hmid
  cdef int iter = 0
  
  hmin = r[1]
  while iter < 20 and addweight(r, w, hmin, NGB) >= mean:
    hmin *= 0.5
    iter = iter + 1

  if iter == 20:
    with gil:
      warn('hmin determination failed %g' % mean)
    return hmin

  iter = 0
  hmax = r[NGB-1]
  while iter < 20 and addweight(r, w, hmax, NGB) <= mean:
    hmax *= 2
    iter = iter + 1

  if iter == 20:
    with gil:
      warn('hmax determination failed %g' % mean)
    return hmax

  iter = 0 
  while iter < 20:
    hmid = (hmin + hmax) * 0.5
    m = addweight(r, w, hmid, NGB)
    if m > mean: hmax = hmid
    elif m < mean: hmin = hmid
    else: break
    if 1 - hmin / hmax < 1e-5: break
    iter = iter + 1

  if iter == 20:
    with gil:
      warn('h determination failed %g' % mean)

  return hmid

@cython.boundscheck(False)
def solve_sml(pos, pweight, locations, weight, out, ztree.Tree tree, int NGB):
    iter = numpy.nditer(
          [pos[..., 0], pos[..., 1], pos[..., 2], pweight, out], 
      op_flags=[['readonly'], ['readonly'], ['readonly'], 
                ['readonly'], ['readwrite']], 
     op_dtypes=['f8', 'f8', 'f8', 'f4', 'f4'],
         flags=['buffered', 'external_loop', 'zerosize_ok'], 
       casting='unsafe')

    cdef npyiter.CIter citer
    cdef size_t size = npyiter.init(&citer, iter)
    cdef double fpos[3]
    cdef double R[3]
    with nogil: 
      while size > 0:
        while size > 0:
          if (<float*>citer.data[4])[0] <= 0:
            for d in range(3):
              fpos[d] = (<double*>citer.data[d])[0]
            tree.get_node_size(tree.get_container(fpos, 0), R)
            (<float*>citer.data[4])[0] = R[0]
          npyiter.advance(&citer)
          size = size - 1
        size = npyiter.next(&citer)

