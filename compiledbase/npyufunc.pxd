cimport cpython
cimport numpy
import numpy

from numpy cimport PyUFuncGenericFunction
from numpy cimport PyUFunc_None
from numpy cimport PyUFunc_One
from numpy cimport PyUFunc_Zero
cdef extern from "numpy/arrayobject.h":
  cdef enum :
    NPY_VERSION

cdef extern from "numpy/ufuncobject.h":
  cdef PyUFuncGenericFunction PyUFunc_dd_d
  cdef PyUFuncGenericFunction PyUFunc_ff_f
  cdef PyUFuncGenericFunction PyUFunc_ff_f_As_dd_d
  cdef PyUFuncGenericFunction PyUFunc_d_d
  cdef PyUFuncGenericFunction PyUFunc_f_f
  cdef PyUFuncGenericFunction PyUFunc_f_f_As_d_d

cdef inline void PyUFunc_dd_d_As_ff_f(char** args, size_t* dimensions, size_t* steps, float (*func)(float , float ) nogil) nogil:
  cdef double * src
  cdef double * op1
  cdef double * dst
  cdef size_t i = 0
  while i < dimensions[0]:
    src = <double*>args[0]
    op1 = <double*>args[1]
    dst = <double*>args[2]
    dst[0] = func(src[0], op1[0])
    args[0] = args[0] + steps[0]
    args[1] = args[1] + steps[1]
    args[2] = args[2] + steps[2]
    i = i + 1


cdef inline void PyUFunc_d_d_As_f_f(char** args, size_t* dimensions, size_t* steps, float (*func)(float) nogil) nogil:
  cdef double * src
  cdef double * dst
  cdef size_t i = 0
  while i < dimensions[0]:
    src = <double*>args[0]
    dst = <double*>args[1]
    dst[0] = func(src[0])
    args[0] = args[0] + steps[0]
    args[1] = args[1] + steps[1]
    i = i + 1

from numpy cimport NPY_FLOAT, NPY_DOUBLE
from numpy cimport PyUFunc_RegisterLoopForType
from numpy cimport PyUFunc_FromFuncAndDataAndSignature
from numpy cimport PyUFunc_FromFuncAndData
from libc.stdint cimport *

cdef struct UFunc2FD:
  void * funcs[2]
  PyUFuncGenericFunction gfs[2]
  char types[10] # anynumber will do

cdef inline register(namespace,
    void * floatfunc,
    void * doublefunc,
    int nin,
    char * identity,
    name, docstring):

    cdef int I = {
      'reorderablenone': -2, #ReorderableNone
      'none': PyUFunc_None,
      'one': PyUFunc_One,
      'zero': PyUFunc_Zero,
    }[identity]
    cdef int nout = 1
    cdef int nall = nin + nout

    cdef int nfuncs = 2
    
    cdef numpy.ndarray bytes = numpy.empty(shape=1, 
          dtype=numpy.dtype([('funcs', (numpy.intp, 2)),
                 ('types', (numpy.int8, (2,nall))),
                 ('wrappers', (numpy.intp, 2))], align=True))
    cdef UFunc2FD  * stru = <UFunc2FD*> bytes.data

    cdef types = [NPY_FLOAT, NPY_DOUBLE]
    cdef int i, j
    for i in range(nall):
      for j in range(nfuncs):
        stru.types[i + nall * j] = types[j]

    stru.funcs[0] = floatfunc
    stru.funcs[1] = doublefunc

    if nin == 2:
      stru.gfs[0] = <PyUFuncGenericFunction>PyUFunc_ff_f
      stru.gfs[1] = <PyUFuncGenericFunction>PyUFunc_dd_d

      if floatfunc == NULL:
        stru.gfs[0] = <PyUFuncGenericFunction>PyUFunc_ff_f_As_dd_d
        stru.funcs[0] = doublefunc

      if doublefunc == NULL:
        stru.gfs[1] = <PyUFuncGenericFunction>PyUFunc_dd_d_As_ff_f
        stru.funcs[1] = floatfunc

    elif nin == 1:
      stru.gfs[0] = <PyUFuncGenericFunction>PyUFunc_f_f
      stru.gfs[1] = <PyUFuncGenericFunction>PyUFunc_d_d

      if floatfunc == NULL:
        stru.gfs[0] = <PyUFuncGenericFunction>PyUFunc_f_f_As_d_d
        stru.funcs[0] = doublefunc
      if doublefunc == NULL:
        stru.gfs[1] = <PyUFuncGenericFunction>PyUFunc_d_d_As_f_f
        stru.funcs[1] = floatfunc

    ufunc = PyUFunc_FromFuncAndData(stru.gfs, stru.funcs, stru.types, nfuncs, nin, nout, I, name, docstring, 0)
    namespace[name] = ufunc
    namespace["__" + name + "_interface__"] = bytes
