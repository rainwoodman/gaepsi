import numpy
cimport cpython
cimport numpy
from libc.stdint cimport *

cdef packed struct NodeInfo:
  int64_t key # from key and level to derive the bot and top limits
  short order
  short child_length
  int parent
  intptr_t first
  intptr_t npar
  int child[8] # child[0]  may save first_par and child[1] may save npar

cdef class Scale(object):
  """Scale scales x,y,z to 0 ~ (1<<bits) - 1 """
  cdef float * _min
  cdef float * _norm
  cdef readonly int bits
  cdef readonly numpy.ndarray min
  cdef readonly numpy.ndarray norm
  cdef void BBint(Scale self, float pos[3], float r, int32_t center[3], int32_t min[3], int32_t max[3]) nogil
  cdef float dist2(Scale self, int32_t center[3], int32_t point[3]) nogil
  cdef void decode(Scale self, int64_t key, int32_t point[3]) nogil
  cdef void decode_float(Scale self, int64_t key, float pos[3]) nogil
  cdef int64_t encode(Scale self, int32_t point[3]) nogil
  cdef int64_t encode_float (Scale self, float pos[3]) nogil
  cdef void from_float(Scale self, float pos[3], int32_t point[3]) nogil

cdef class Result:
  cdef intptr_t * _buffer
  cdef float * _weight
  cdef readonly size_t used
  cdef readonly size_t size
  cdef readonly size_t limit
  cdef void _grow(Result self) nogil
  cdef void truncate(Result self) nogil
  cdef void append_one(Result self, intptr_t i) nogil
  cdef void append_one_with_weight(Result self, intptr_t i, float weight) nogil
  cdef void append(Result self, intptr_t start, intptr_t length) nogil
  cpdef harvest(Result self)

cdef class Tree(object):
  cdef NodeInfo * _buffer
  cdef readonly size_t size
  cdef readonly size_t used
  cdef readonly size_t thresh
  cdef readonly int64_t[:] _zorder
  cdef readonly numpy.ndarray zorder
  cdef readonly Scale scale
  cdef void _grow(self) nogil
  cdef int32_t __query_neighbours_estimate_radius(Tree self, int64_t ckey, int count) nogil
  cdef void __add_node(Tree self, Result result, int32_t min[3], int32_t max[3], int32_t center[3], intptr_t node) nogil
  cdef int __goodness(Tree self, intptr_t node, int32_t min[3], int32_t max[3]) nogil
  cdef void __query_box_one_from(Tree self, Result result, int32_t min[3], int32_t max[3], int32_t center[3], intptr_t root) nogil
  cdef void query_box_one(Tree self, Result result, int32_t min[3], int32_t max[3], int32_t center[3]) nogil
  cdef int _tree_build(Tree self) nogil
  cdef intptr_t _create_child(self, intptr_t first_par, intptr_t parent) nogil
  cdef void query_neighbours_one(Tree self, Result result, float pos[3]) nogil
