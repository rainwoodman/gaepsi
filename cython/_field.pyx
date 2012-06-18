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
cdef float solve_sml_one(float pos[3], float * r, float * w, int NGB, float mean) nogil:
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
    if len(pos) == 0: return
    iter = numpy.nditer(
          [pos[..., 0], pos[..., 1], pos[..., 2], pweight, out], 
      op_flags=[['readonly'], ['readonly'], ['readonly'], 
                ['readonly'], ['readwrite']], 
     op_dtypes=['f4', 'f4', 'f4', 'f4', 'f4'],
         flags=['buffered', 'external_loop'], 
       casting='unsafe')

    cdef numpy.ndarray weights = numpy.atleast_1d(weight)
    cdef numpy.ndarray[numpy.float32_t, ndim=1] r
    cdef numpy.ndarray[numpy.float32_t, ndim=1] w

    cdef float * r_ptr
    cdef float * w_ptr
    cdef float w0 = 0
    cdef float x
    cdef intptr_t i
    cdef float fpos[3]
    cdef float[:, :] _locations = locations
    cdef float[:] _weights = weights

    cdef npyiter.NpyIter * citer = npyiter.GetNpyIter(iter)
    cdef npyiter.IterNextFunc next = npyiter.GetIterNext(citer, NULL)
    cdef char ** data = npyiter.GetDataPtrArray(citer)
    cdef numpy.npy_intp *strides = npyiter.GetInnerStrideArray(citer)
    cdef numpy.npy_intp *size_ptr = npyiter.GetInnerLoopSizePtr(citer)
    cdef intptr_t iop, size
    cdef intptr_t total = 0
    cdef ztree.Result result
    cdef int32_t ib
    cdef uint64_t key
    if NGB > 0:
      result = ztree.Result(NGB)
      r = numpy.empty(shape=NGB, dtype=numpy.float32)
      r_ptr = <float*>r.data
      w = numpy.empty(shape=NGB, dtype=numpy.float32)
      w_ptr = <float*>w.data
    with nogil: 
     while True:
      size = size_ptr[0]
      total += size
      if NGB <= 0:
        while size > 0:
          if (<float*>data[4])[0] <= 0:
            for d in range(3):
              fpos[d] = (<float*>data[d])[0]
            key = tree.zorder.encode_float(fpos)
            ib = tree.query_neighbours_estimate_radius(key, 1)
            (<float*>data[4])[0] = ib / tree.zorder._norm[0]
          for iop in range(5):
            data[iop] += strides[iop]
          size = size - 1
      else:
        while size > 0:
          if (<float*>data[4])[0] <= 0:
            for d in range(3):
              fpos[d] = (<float*>data[d])[0]
            w0 = NGB * (<float*>data[3])[0]
            result.truncate()
            tree.query_neighbours_one(result, fpos)
            for i in range(NGB):
              if _weights.shape[0] > 1:
                w_ptr[i] = _weights[result._buffer[i]]
              else:
                w_ptr[i] = _weights[0]
              r_ptr[i] = 0
              for d in range(3):
                x = _locations[result._buffer[i], d] - fpos[d]
                r_ptr[i] = r_ptr[i] + x * x
              r_ptr[i] = r_ptr[i] ** 0.5
  
            (<float*>data[4])[0] = solve_sml_one(fpos, r_ptr, w_ptr, NGB, w0 / (4 * 3.1416 / 3))
          for iop in range(5):
            data[iop] += strides[iop]
          size = size - 1
      if next(citer) == 0: break

