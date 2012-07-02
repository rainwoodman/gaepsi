cimport cpython
cimport numpy
import numpy

from numpy cimport PyUFuncGenericFunction
from numpy cimport PyUFunc_None
cdef extern from "numpy/arrayobject.h":
  cdef enum :
    NPY_VERSION

from numpy cimport NPY_FLOAT, NPY_DOUBLE
from numpy cimport PyUFunc_RegisterLoopForType
from numpy cimport PyUFunc_FromFuncAndDataAndSignature
from numpy cimport PyUFunc_FromFuncAndData
from numpy cimport PyUFunc_dd_d, PyUFunc_ff_f
from libc.stdint cimport *

cdef struct UFunc2to1:
  void * funcs[2]
  char types[6]
  PyUFuncGenericFunction gfs[2]

cdef inline register(namespace,
    float (*ff_f)(float, float) nogil, 
    double (*dd_d)(double, double) nogil,
    name, docstring):

    # this thing is added recently in 1.7.
    cdef int PyUFunc_ReorderableNone = -2
    
    cdef numpy.ndarray bytes = numpy.empty(shape=1, 
          dtype=[('funcs', (numpy.intp, 2)),
                 ('types', (numpy.int8, (2,3))),
                 ('wrappers', (numpy.intp, 2))])
    cdef UFunc2to1 * stru = <UFunc2to1*> bytes.data

    stru.types[0] = NPY_FLOAT
    stru.types[1] = NPY_FLOAT
    stru.types[2] = NPY_FLOAT
    stru.types[3] = NPY_DOUBLE
    stru.types[4] = NPY_DOUBLE
    stru.types[5] = NPY_DOUBLE
    stru.gfs[0] = PyUFunc_ff_f
    stru.gfs[1] = PyUFunc_dd_d
    stru.funcs[0] = <void*>ff_f
    stru.funcs[1] = <void*>dd_d
    ufunc = PyUFunc_FromFuncAndData(stru.gfs, stru.funcs, stru.types, 2, 2, 1, PyUFunc_ReorderableNone, name, docstring, 0)
    namespace[name] = ufunc
    namespace["__" + name + "_interface__"] = bytes
