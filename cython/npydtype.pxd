cimport numpy
from cpython.object cimport PyObject, PyTypeObject
from cpython.ref cimport Py_XINCREF
from libc.stdint cimport intptr_t
cdef extern from *:
  cdef enum:
    NPY_USE_GETITEM
    NPY_USE_SETITEM

  ctypedef struct PyArray_ArrFuncs:
    void *cast[0]
    void *getitem
    void *setitem
    void *copyswapn
    void *copyswap
    void *compare
    void *argmax
    void *dotfunc
    void *scanfunc
    void *fromstr
    void *nonzero
    void *fill
    void *fillwithscalar
    void * sort[3]
    void * argsort[3]
    PyObject * castdict
    void * scalarkind
    int **cancastscalarkindto
    int *cancastto
    int listpickle
    void *argmin

  ctypedef struct PyArray_ArrayDescr:
    void * base
    void * shape

  int PyArray_RegisterDataType (PyArray_Descr * descr) except -1
  void PyArray_InitArrFuncs (PyArray_ArrFuncs * funcs)
  int PyArray_RegisterCastFunc (numpy.dtype descr, int totype, void * castfunc)
  int PyArray_RegisterCanCast(numpy.dtype descr, int totype, int scalar_kind)
  int PyUFunc_RegisterLoopForType(numpy.ufunc ufunc, int usertype, void * function, int* arg_types, void* data) except *

  ctypedef struct PyUFuncObject:
    PyObject * userloops
    void ** functions
    void ** data
    int ntypes
    char *types

  ctypedef struct PyArray_Descr:
    PyObject * typeobj
    char kind
    char type
    char byteorder
    int flags
    int type_num
    int elsize
    int alignment
    PyObject * metadata
    PyArray_ArrayDescr * subarray
    PyObject * fields
    PyArray_ArrFuncs * f

cdef inline void register_safe_castfuncs(int typenum, object list) except *:
   """ list is a dictionary from dtype str to cast func,
       register the list of types that typenum can be safely cast from. """
   cdef intptr_t func
   cdef object dtype
   for key in list:
     dtype = numpy.dtype(key)
     func = <intptr_t> list[key]
     PyArray_RegisterCastFunc (dtype, typenum, <void*>func)
     if key == 'object':
       PyArray_RegisterCanCast  (dtype, typenum, numpy.NPY_OBJECT_SCALAR)
     else:
       PyArray_RegisterCanCast  (dtype, typenum, numpy.NPY_NOSCALAR)

cdef inline int register_dtype(PyArray_Descr * descr, PyArray_ArrFuncs * f,
     object dict) except *:
  (<PyObject*> (descr)).ob_type = <PyTypeObject*> numpy.dtype
  (<PyObject*> (descr)).ob_refcnt = 1
  descr.typeobj = <PyObject*>dict['typeobj']
  Py_XINCREF(descr.typeobj)
  descr.subarray = NULL
  descr.fields = NULL
  descr.elsize = dict['elsize']
  descr.kind = ord(dict['kind'])
  descr.byteorder = ord(dict['byteorder'])
  descr.type = ord(dict['type'])
  descr.alignment = dict['alignment']
  descr.metadata = <PyObject*>dict['metadata']
  Py_XINCREF(descr.metadata)
  descr.f = f
  return PyArray_RegisterDataType(descr)

cdef inline void register_ufuncs(int typenum, void* func, object types, object list) except *:
   """ list is a dictionary from numpy ufunc to data passed to func
       types is a list of integer typenums. """
   cdef intptr_t data
   cdef numpy.ndarray typesarray = numpy.array(types, dtype='i4')
   for ufunc in list:
     data = <intptr_t> ord(list[ufunc])
     PyUFunc_RegisterLoopForType(ufunc, typenum, func, <int*> typesarray.data, <void*> data)
  
