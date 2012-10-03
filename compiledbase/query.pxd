from ztree cimport Tree, node_t, TreeIter
from libc.stdint cimport *
import numpy
cimport numpy
cimport flexarray
cimport npyiter

ctypedef struct Element:
  intptr_t index
  double weight

cdef class Scratch:
  cdef flexarray.FlexArray fa
  cdef readonly numpy.dtype dtype

  cdef inline void reset(self) nogil:
    self.fa.used = 0

  cdef inline void set_cmpfunc(self, flexarray.cmpfunc cmpfunc):
    self.fa.cmpfunc = cmpfunc

  cdef inline void * get_ptr(self, intptr_t index) nogil:
    return flexarray.get_ptr(&self.fa, index)
    
  cdef void add_item(self, void * data) nogil

cdef class Query:
  cdef readonly Tree tree
  cdef readonly numpy.dtype dtype
  cdef readonly size_t sizehint
  cdef public node_t root
  # we do not temper with the cmpfunc in dtype. it's not worth the effort
  cdef flexarray.cmpfunc cmpfunc
  # called by subclasses
  cdef inline void set_cmpfunc(self, flexarray.cmpfunc cmpfunc):
    self.cmpfunc = cmpfunc

  cdef tuple _iterover(self, variables, dtypes, flags)
  # to be overidden
  cdef void execute(self, TreeIter iter, Scratch scratch, char** data) nogil


