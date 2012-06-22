#cython: embedsignature=True
#cython: cdivision=True
import numpy
cimport cpython
import cython
cimport cython
cimport numpy
from libc.stdint cimport *
from libc.stdlib cimport malloc, realloc, free
from zorder cimport Zorder

cdef packed struct NodeInfo:
  int64_t key # from key and level to derive the bot and top limits
  short order
  short child_length
  int parent
  intptr_t first
  intptr_t npar
  int child[8] # child[0]  may save first_par and child[1] may save npar

cdef class Result:
  cdef intptr_t * _buffer
  cdef float * _weight
  cdef readonly size_t used
  cdef readonly size_t size
  cdef readonly size_t limit
  cdef inline void grow(Result self) nogil:
    if self.size < 1024576:
      self.size *= 2
    else:
      self.size += 1024576
    self._buffer = <intptr_t *>realloc(self._buffer, sizeof(intptr_t) * self.size)
    if self.limit > 0:
      self._weight = <float *>realloc(self._buffer, sizeof(float) * self.size)

  cdef inline void truncate(Result self) nogil:
    self.used = 0

  cdef inline void append_one(Result self, intptr_t i) nogil:
    if self.size - self.used <= 1:
      self.grow()
    self._buffer[self.used] = i
    self.used = self.used + 1
  cdef inline void append_one_with_weight(Result self, intptr_t i, float weight) nogil:
    cdef intptr_t k
    if self.used == self.limit:
      if weight >= self._weight[self.used - 1]: return
    else:
      self.used = self.used + 1
    k = self.used - 1
    while k > 0 and weight < self._weight[k - 1]:
      self._weight[k] = self._weight[k - 1]
      self._buffer[k] = self._buffer[k - 1]
      k = k - 1
    self._weight[k] = weight
    self._buffer[k] = i
  cdef inline void append(Result self, intptr_t start, intptr_t length) nogil:
    cdef intptr_t i
    while self.size - self.used <= length:
      self.grow()
    for i in range(start, start + length):
      self._buffer[self.used] = i
      self.used = self.used + 1
  cdef inline numpy.ndarray harvest(Result self):
    cdef cython.view.array array
    array = <intptr_t[:self.used]> self._buffer
    array.callback_free_data = free
    self._buffer = NULL
    return numpy.asarray(array)

cdef class Tree:
  cdef NodeInfo * _buffer
  cdef readonly size_t size
  cdef readonly size_t used
  cdef readonly size_t thresh
  cdef int64_t * _zkey
  cdef size_t _zkey_length
  cdef readonly numpy.ndarray zkey
  cdef readonly Zorder zorder

  cdef void get_node_pos(Tree self, intptr_t index, float pos[3]) nogil
  cdef float get_node_size(Tree self, intptr_t index) nogil
  cdef intptr_t get_container(Tree self, float pos[3], int atleast) nogil

  cdef inline void _grow(Tree self) nogil:
    if self.size < 1024576 * 16:
      self.size *= 2
    else:
      self.size += 1024576 * 16
    self._buffer = <NodeInfo * >realloc(self._buffer, sizeof(NodeInfo) * self.size)
  cdef int32_t _estimate_radius(Tree self, float pos[3], int count) nogil
  cdef void __add_node(Tree self, Result result, int32_t min[3], int32_t max[3], int32_t center[3], intptr_t node) nogil
  cdef int __goodness(Tree self, intptr_t node, int32_t min[3], int32_t max[3]) nogil
  cdef void __query_box_one_from(Tree self, Result result, int32_t min[3], int32_t max[3], int32_t center[3], intptr_t root) nogil
  cdef void query_box_one(Tree self, Result result, int32_t min[3], int32_t max[3], int32_t center[3]) nogil
  cdef int _tree_build(Tree self) nogil
  cdef intptr_t _create_child(self, intptr_t first_par, intptr_t parent) nogil
  cdef void query_neighbours_one(Tree self, Result result, float pos[3]) nogil
