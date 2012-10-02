#cython: embedsignature=True
#cython: cdivision=True
import numpy
cimport numpy
from ztree cimport Tree, node_t
from libc.stdint cimport *
cimport npyiter
from libc.math cimport sqrt
cimport flexarray

numpy.import_array()

cdef int elecmpfunc(Element * e1, Element * e2) nogil:
  return (e1.weight < e2.weight) - (e1.weight > e2.weight)

cdef class ResultSet:
  property used:
    def __get__(self): return self.fa.used

  property array:
    def __get__(self): return flexarray.tonumpy(&self.fa, [('indices', 'i8'), ('weights', 'f8')], self)

  property indices:
    def __get__(self): return self.array['indices']
  property weights:
    def __get__(self): return self.array['weights']

  def __cinit__(self, int size):
    flexarray.init(&self.fa, <void**>&self._e, sizeof(Element), size)
    self.fa.cmpfunc = <flexarray.cmpfunc> elecmpfunc

  def __dealloc__(self):
    flexarray.destroy(&self.fa)

cdef class Query:

  def __cinit__(self):
    flexarray.init(&self.indices, NULL, sizeof(intptr_t), 1024)

  def __init__(self, tree, size):
    self.tree = tree
    self.resultset = ResultSet(size)

  def __dealloc__(self):
    flexarray.destroy(&self.indices)

  cdef void execute(self, char** data) nogil:
    pass
       
