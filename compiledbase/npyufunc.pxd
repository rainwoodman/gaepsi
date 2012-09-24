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
  cdef PyUFuncGenericFunction PyUFunc_d_d
  cdef PyUFuncGenericFunction PyUFunc_f_f
  cdef PyUFuncGenericFunction PyUFunc_ddd_d
  cdef PyUFuncGenericFunction PyUFunc_fff_f

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
    
    cdef numpy.ndarray bytes = numpy.empty(shape=1, 
          dtype=numpy.dtype([('funcs', (numpy.intp, 2)),
                 ('types', (numpy.int8, (2,nall))),
                 ('wrappers', (numpy.intp, 2))], align=True))
    cdef UFunc2FD  * stru = <UFunc2FD*> bytes.data

    cdef int i
    for i in range(nin):
      stru.types[i] = NPY_FLOAT
      stru.types[nall + i] = NPY_DOUBLE
    stru.types[nin] = NPY_FLOAT
    stru.types[nall + nin] = NPY_DOUBLE

    if nin == 2:
      stru.gfs[0] = PyUFunc_ff_f
      stru.gfs[1] = PyUFunc_dd_d
    elif nin == 1:
      stru.gfs[0] = PyUFunc_f_f
      stru.gfs[1] = PyUFunc_d_d

    stru.funcs[0] = floatfunc
    stru.funcs[1] = doublefunc
    ufunc = PyUFunc_FromFuncAndData(stru.gfs, stru.funcs, stru.types, 2, nin, nout, I, name, docstring, 0)
    namespace[name] = ufunc
    namespace["__" + name + "_interface__"] = bytes
