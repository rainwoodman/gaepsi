from libc.stdint cimport *
from ztree cimport Tree, node_t
from fillingcurve cimport fckey_t
cdef class Query:
  cdef readonly size_t used
  cdef readonly size_t size
  cdef readonly size_t limit
  cdef readonly fckey_t centerkey
  cdef fckey_t AABBkey[2]
  cdef intptr_t * _items
  cdef double * _weight
  cdef int _weighted

  cdef intptr_t * steal(Query self) nogil
  cdef void execute_one(Query self, Tree tree, double pos[3], double size[3]) nogil
  cdef void execute_r(Query self, Tree tree, node_t node) nogil
  cdef void _add_node_straight(self, Tree tree, node_t node) nogil
  cdef void _add_node_weighted(self, Tree tree, node_t node) nogil
  cdef void _add_node_lcn(self, Tree tree, node_t node) nogil
  cdef void raytrace_one_r(Query self, Tree tree, node_t node, double p0[3], double dir[3], double tE, double tL) nogil
  cdef void raytrace_lcn_r(Query self, Tree tree, node_t node, double p0[3], double dir[3], double tE, double tL) nogil
