cimport numpy
from cpython.ref cimport PyObject

ctypedef void * NpyIter

ctypedef int (*IterNextFunc)(NpyIter * iter) nogil
ctypedef void (*GetMultiIndexFunc)(NpyIter * iter, numpy.npy_intp *outcoords) nogil

ctypedef struct NewNpyArrayIterObject:
    PyObject base
    NpyIter *iter

cdef inline NpyIter* GetNpyIter(object iter):
  return (<NewNpyArrayIterObject*>iter).iter

cdef extern from "numpy/arrayobject.h":
  IterNextFunc GetIterNext "NpyIter_GetIterNext" (NpyIter *iter, char **)
  char** GetDataPtrArray "NpyIter_GetDataPtrArray" (NpyIter* iter)
  numpy.npy_intp * GetInnerStrideArray "NpyIter_GetInnerStrideArray" (NpyIter*  iter)
  numpy.npy_intp * GetInnerLoopSizePtr "NpyIter_GetInnerLoopSizePtr" (NpyIter* iter)
  int GetNDim "NpyIter_GetNDim" (NpyIter* iter)
  int GetNOp "NpyIter_GetNOp" (NpyIter* iter)
  numpy.npy_bool IsBuffered "NpyIter_IsBuffered" (NpyIter * iter)
  numpy.npy_intp GetBufferSize "NpyIter_GetBufferSize" (NpyIter * iter)


