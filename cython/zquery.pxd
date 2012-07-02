from libc.stdint cimport *
from ztree cimport Tree

cdef class Query:
  cdef readonly size_t used
  cdef readonly size_t size
  cdef readonly size_t limit
  cdef readonly int64_t centerkey
  cdef int64_t AABBkey[2]
  cdef intptr_t * _items
  cdef float * _weight
  cdef int _weighted

  cdef intptr_t * steal(Query self) nogil
  cdef void execute_one(Query self, Tree tree, double pos[3], double size[3]) nogil
  cdef void execute_r(Query self, Tree tree, intptr_t node) nogil
  cdef void _add_node_straight(self, Tree tree, intptr_t node) nogil
  cdef void _add_node_weighted(self, Tree tree, intptr_t node) nogil
