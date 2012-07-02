#cython: embedsignature=True
#cython: cdivision=True
import numpy
cimport cpython
import cython
cimport cython
cimport numpy
from libc.stdint cimport *
from libc.stdlib cimport malloc, realloc, free
cimport zorder
from zorder cimport zorder_t

cdef packed struct NodeInfo:
  zorder_t key # from key and level to derive the bot and top limits
  short order
  short child_length
  int parent
  intptr_t first
  intptr_t npar
  int child[8] # child[0]  may save first_par and child[1] may save npar

cdef class Tree:
  cdef NodeInfo * _nodes
  cdef readonly size_t size
  cdef readonly size_t used
  cdef readonly size_t thresh
  cdef zorder_t * _zkey
  cdef size_t _zkey_length
  cdef readonly numpy.ndarray zkey
  cdef readonly zorder.Digitize digitize

  cdef inline void get_node_pos(Tree self, intptr_t index, double pos[3]) nogil:
    cdef int32_t ipos[3]
    zorder.decode(self._nodes[index].key, ipos)
    self.digitize.i2f(ipos, pos)

  cdef inline void get_leaf_pos(Tree self, intptr_t index, double pos[3]) nogil:
    cdef int32_t ipos[3]
    zorder.decode(self._zkey[index], ipos)
    self.digitize.i2f(ipos, pos)

  cdef inline void get_node_size(Tree self, intptr_t index, double size[3]) nogil:
    cdef int32_t isize[3]
    isize[0] = ((1<<(self._nodes[index].order+1)) - 1)
    isize[1] = isize[0]
    isize[2] = isize[0]
    self.digitize.i2f0(isize, size)

  cdef inline intptr_t get_container(Tree self, double pos[3], int atleast) nogil:
    cdef zorder_t key
    cdef int32_t ipos[3]
    self.digitize.f2i(pos, ipos)
    key = zorder.encode(ipos)
    return self.get_container_key(key, atleast)

  cdef inline intptr_t get_container_key(Tree self, zorder_t key, int atleast) nogil:
    cdef intptr_t this, child, next
    this = 0
    while this != -1 and self._nodes[this].child_length > 0:
      next = this
      for i in range(self._nodes[this].child_length):
        child = self._nodes[this].child[i]
        if zorder.boxtest(self._nodes[child].key, self._nodes[child].order, key):
          next = child
          break
      if next == this: break
      else:
        if self._nodes[next].npar < atleast: break
        this = next
        continue
    return this

  cdef inline void _grow(Tree self) nogil:
    if self.size < 1024576 * 16:
      self.size *= 2
    else:
      self.size += 1024576 * 16
    self._nodes = <NodeInfo * >realloc(self._nodes, sizeof(NodeInfo) * self.size)

  cdef int _tree_build(Tree self) nogil
  cdef intptr_t _create_child(self, intptr_t first_par, intptr_t parent) nogil

