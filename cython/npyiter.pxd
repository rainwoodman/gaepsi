cimport numpy
"""
  Example:
  cdef void unit_of_work(float * in, float * out) nogil:
    out[0] = in[0] * 2
   
  def work(in, out):
    iter = numpy.nditer(
          [in, out, ], 
      op_flags=[['readonly'], ['writeonly']],
     op_dtypes=['f4', 'f4'],
         flags=['buffered', 'external_loop'], 
       casting='unsafe')
    cdef npyiter.CIter citer
    cdef size_t size = npyiter.init(&citer, iter)
    with nogil:
      while size > 0:
        while size > 0:
          unit_of_work(<float*>citer.data[0],
             <float*>citer.data.data[1])
          npyiter.advance(&citer)
          size = size - 1
        size = npyiter.next(&citer)
"""
from cpython.ref cimport PyObject

ctypedef void NpyIter
ctypedef int (*IterNextFunc)(NpyIter * iter) nogil
ctypedef void (*GetMultiIndexFunc)(NpyIter * iter, numpy.npy_intp *outcoords) nogil

ctypedef struct NewNpyArrayIterObject:
    PyObject base
    NpyIter *iter

cdef extern from "numpy/arrayobject.h":
  IterNextFunc GetIterNext "NpyIter_GetIterNext" (NpyIter *iter, char **)
  char** GetDataPtrArray "NpyIter_GetDataPtrArray" (NpyIter* iter)
  numpy.npy_intp * GetInnerStrideArray "NpyIter_GetInnerStrideArray" (NpyIter*  iter)
  numpy.npy_intp * GetInnerLoopSizePtr "NpyIter_GetInnerLoopSizePtr" (NpyIter* iter)
  int GetNDim "NpyIter_GetNDim" (NpyIter* iter)
  int GetNOp "NpyIter_GetNOp" (NpyIter* iter)
  numpy.npy_bool IsBuffered "NpyIter_IsBuffered" (NpyIter * iter)
  numpy.npy_intp GetBufferSize "NpyIter_GetBufferSize" (NpyIter * iter)

cdef inline NpyIter* GetNpyIter(object iter):
  return (<NewNpyArrayIterObject*>iter).iter

ctypedef struct CIter:
  NpyIter * npyiter
  IterNextFunc _next
  char ** data
  numpy.npy_intp * strides
  numpy.npy_intp * size_ptr
  int nop

cdef inline size_t next(CIter* self) nogil:
  if self._next(self.npyiter) == 0:
    return 0
  return self.size_ptr[0]

cdef inline size_t init(CIter * self, iter):
    self.npyiter = GetNpyIter(iter)
    self._next = <IterNextFunc>GetIterNext(self.npyiter, NULL)
    self.data = GetDataPtrArray(self.npyiter)
    self.strides = GetInnerStrideArray(self.npyiter)
    self.size_ptr = GetInnerLoopSizePtr(self.npyiter)
    self.nop = GetNOp(self.npyiter)
    return self.size_ptr[0]

cdef inline void advance(CIter * self) nogil:
    cdef int iop
    for iop in range(self.nop):
      self.data[iop] += self.strides[iop]
