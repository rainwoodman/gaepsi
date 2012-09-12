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
from zorder cimport zorder_t, ipos_t
ctypedef int node_t
cimport flexarray
from flexarray cimport FlexArray

cdef packed struct Node:
  zorder_t key # from key and level to derive the bot and top limits
  intptr_t first 
  intptr_t npar
  node_t parent # if parent is -1, this node is unused/reclaimable
  short order
  short child_length
  node_t child[8] # child[0]  may save first_par and child[1] may save npar

cdef packed struct LeafNode:
  zorder_t key # from key and level to derive the bot and top limits
  intptr_t first 
  intptr_t npar
  node_t parent # if parent is -1, this node is unused/reclaimable.
  short order
  short child_length # padding, shall aways be 0

cdef inline node_t mark_leaf(node_t index) nogil:
  return index | (<node_t>1 << (sizeof(node_t) * 8 - 1))

cdef class TreeNode:
  cdef readonly node_t _index
  cdef readonly Tree tree

cdef class Tree:
  cdef Node * nodes
  cdef LeafNode * leafnodes
  cdef FlexArray _nodes
  cdef FlexArray _leafnodes
  cdef readonly size_t minthresh
  cdef readonly size_t maxthresh
  cdef zorder_t * _zkey
  cdef size_t _zkey_length
  cdef readonly numpy.ndarray zkey
  cdef readonly zorder.Digitize digitize

  cdef inline void * get_node_pointer(Tree self, node_t index) nogil:
    if index & (<node_t>1 << (sizeof(node_t) * 8 - 1)):
      return &self.leafnodes[index & ~(<node_t>1<<(sizeof(node_t) * 8 - 1))]
    elif index >= self._nodes.used:
      return &self.leafnodes[index - self._nodes.used]
    else:
      return &self.nodes[index]

  cdef inline bint is_leafnode(Tree self, node_t index) nogil:
    if index & (<node_t>1 << (sizeof(node_t) * 8 - 1)):
      return True
    elif index >= self._nodes.used:
      return True
    return False

  cdef inline node_t leafnode_index(Tree self, node_t index) nogil:
    if index & (<node_t>1 << (sizeof(node_t) * 8 - 1)):
      return index & ~(<node_t>1<<(sizeof(node_t) * 8 - 1))
    elif index >= self._nodes.used:
      return index - self._nodes.used
    else:
      return -1

  cdef inline node_t node_index(Tree self, node_t index) nogil:
    if index & (<node_t>1 << (sizeof(node_t) * 8 - 1)):
      return (index & ~(<node_t>1<<(sizeof(node_t) * 8 - 1))) + self._nodes.used
    else:
      return index

  cdef inline size_t get_length(Tree self) nogil:
    return self._nodes.used + self._leafnodes.used

  cdef inline bint get_node_reclaimable(Tree self, node_t index) nogil:
    return (index != 0) and (self.get_node_parent(index) == -1)

  cdef inline void get_node_pos(Tree self, node_t index, double pos[3]) nogil:
    """ returns the topleft corner of the node """
    cdef ipos_t ipos[3]
    zorder.decode(self.get_node_key(index), ipos)
    self.digitize.i2f(ipos, pos)

  cdef inline void get_leaf_pos(Tree self, node_t index, double pos[3]) nogil:
    cdef ipos_t ipos[3]
    zorder.decode(self._zkey[index], ipos)
    self.digitize.i2f(ipos, pos)

  cdef inline size_t get_node_npar(Tree self, node_t index) nogil:
    return (<LeafNode*>self.get_node_pointer(index))[0].npar

  cdef inline void set_node_npar(Tree self, node_t index, size_t npar) nogil:
    (<LeafNode*>self.get_node_pointer(index))[0].npar = npar

  cdef inline intptr_t get_node_first(Tree self, node_t index) nogil:
    return (<LeafNode*>self.get_node_pointer(index))[0].first

  cdef inline zorder_t get_node_key(Tree self, node_t index) nogil:
    return (<LeafNode*>self.get_node_pointer(index))[0].key

  cdef inline int get_node_order(Tree self, node_t index) nogil:
    return (<LeafNode*>self.get_node_pointer(index))[0].order

  cdef inline node_t get_node_parent(Tree self, node_t index) nogil:
    return (<LeafNode*>self.get_node_pointer(index))[0].parent

  cdef inline size_t get_node_nchildren(Tree self, node_t index) nogil:
    return (<LeafNode*>self.get_node_pointer(index))[0].child_length

  cdef inline node_t * get_node_children(Tree self, node_t index, int * count) nogil:
    count[0] = self.get_node_nchildren(index)
    if self.is_leafnode(index): return NULL
    return self.nodes[index].child

  cdef inline void get_node_size(Tree self, node_t index, double size[3]) nogil:
    cdef ipos_t isize[3]
    isize[0] = (<ipos_t>1<<(self.get_node_order(index))) - 1
    isize[1] = isize[0]
    isize[2] = isize[0]
    self.digitize.i2f0(isize, size)

  cdef inline node_t get_container(Tree self, double pos[3], int atleast) nogil:
    cdef zorder_t key
    cdef ipos_t ipos[3]
    self.digitize.f2i(pos, ipos)
    key = zorder.encode(ipos)
    return self.get_container_by_key(key, atleast)

  cdef inline node_t get_container_by_key(Tree self, zorder_t key, int atleast) nogil:
    cdef node_t this, child, next
    this = 0
    cdef int nchildren
    cdef node_t * children
    children = self.get_node_children(this, &nchildren)
    while this != -1 and nchildren > 0:
      next = this
      for i in range(nchildren):
        if zorder.boxtest(self.get_node_key(children[i]), self.get_node_order(children[i]), key):
          next = children[i]
          break
      if next == this: break
      else:
        if self.get_node_npar(next) < atleast: break
        this = next
        continue
    return this

  cdef int _tree_build(Tree self) nogil except -1
  cdef intptr_t _optimize(Tree self) nogil except -1
  cdef node_t _create_child(self, intptr_t first_par, intptr_t parent) nogil except -1
  cdef node_t _try_merge_children(self, intptr_t parent) nogil except -1
  cdef node_t _split_node(self, intptr_t node) nogil except -1

