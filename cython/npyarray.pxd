#cython: cdivision=True
cimport numpy
cimport cython

from libc.stdint cimport *
cdef extern from 'numpy/arrayobject.h':
    ctypedef void * PyArray_CompareFunc
    ctypedef void * PyArray_ArgSortFunc
    ctypedef void * PyArray_VectorUnaryFunc

    ctypedef struct PyArray_ArrFuncs:
      PyArray_CompareFunc * compare
      PyArray_ArgSortFunc ** argsort
      PyArray_VectorUnaryFunc ** cast
      int * cancastto

    ctypedef class numpy.dtype [object PyArray_Descr]:
        cdef int type_num
        cdef int itemsize "elsize"
        cdef char byteorder
        cdef object fields
        cdef tuple names
        cdef PyArray_ArrFuncs *f


ctypedef void (*castfunc)(void * , void *, intptr_t, void * , void *) nogil
ctypedef struct CArray:
  char * data
  int itemsize
  int ndim
  numpy.npy_intp * strides
  numpy.npy_intp * shape
  numpy.npy_intp size
  castfunc * castfunc

cdef inline void init(CArray * array, numpy.ndarray obj):
  array.data = obj.data
  array.ndim = obj.ndim
  array.strides = obj.strides
  array.shape = obj.shape
  array.itemsize = obj.descr.itemsize
  cdef int i
  array.size = 1
  for i in range(array.ndim):
    array.size = array.size * array.shape[i]
  array.castfunc = <castfunc*> (<dtype>(obj.descr)).f.cast

cdef inline intptr_t _(CArray * array, intptr_t * i, int n) nogil:
  cdef int d
  cdef intptr_t ret = 0
  for d in range(n):
    ret += (i[d] % array.shape[d]) * array.strides[d]
  return ret

cdef inline void flat(CArray * array, intptr_t ind, cython.numeric * ptr) nogil:
  get(array, (ind % array.size)* array.itemsize, ptr)

cdef inline int _typenum(cython.numeric *ptr) nogil:
  if cython.numeric is cython.float:
    return numpy.NPY_FLOAT32
  elif cython.numeric is cython.double:
    return numpy.NPY_FLOAT64
  elif cython.numeric is cython.int:
    return numpy.NPY_INT32
  elif cython.numeric is cython.long:
    return numpy.NPY_INT64
  elif cython.numeric is cython.short:
    return numpy.NPY_INT16
  else:
    return numpy.NPY_INT64
#    with gil: raise TypeError("do not access type %s", cython.typeof(ptr[0]))
  
cdef inline void get(CArray * array, intptr_t offset, cython.numeric * ptr) nogil:
  array.castfunc[_typenum(ptr)](array.data + offset, ptr, 1, NULL, NULL)


