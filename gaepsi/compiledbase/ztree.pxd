#cython: embedsignature=True
#cython: cdivision=True
import numpy
cimport cpython
import cython
cimport cython
cimport numpy
from libc.stdint cimport *
from libc.stdlib cimport malloc, realloc, free
cimport fillingcurve
from fillingcurve cimport fckey_t, ipos_t
from libc.stdlib cimport abort
ctypedef int node_t
cimport flexarray
from flexarray cimport FlexArray
cimport npyarray

cdef packed struct Node:
  fckey_t key # from key and level to derive the bot and top limits
  intptr_t first 
  intptr_t npar
  node_t parent # if parent is -1, this node is unused/reclaimable
  char order
  char complete
  short child_length
  node_t child[8] # child[0]  may save first_par and child[1] may save npar

cdef packed struct LeafNode:
  fckey_t key # from key and level to derive the bot and top limits
  intptr_t first 
  intptr_t npar
  node_t parent # if parent is -1, this node is unused/reclaimable.
  char order
  char complete
  short child_length # padding, shall aways be 0

cdef inline node_t mark_leaf(node_t index) nogil:
  """ mark a linear leaf node index as a leaf node index,
      setting the highest bit, basically"""
  return index | (<node_t>1 << (sizeof(node_t) * 8 - 1))

cdef class TreeNode:
  cdef readonly node_t _index
  cdef readonly Tree tree

cdef class Tree:
  cdef object __weakref__
  cdef Node * nodes
  cdef LeafNode * leafnodes
  cdef FlexArray _nodes
  cdef FlexArray _leafnodes
  cdef readonly size_t minthresh
  cdef readonly size_t maxthresh
  cdef fckey_t * _zkey
  cdef intptr_t * _arg
  cdef size_t _zkey_length
  cdef readonly numpy.ndarray zkey
  cdef readonly numpy.ndarray scale
  cdef readonly numpy.ndarray arg
  cdef double * _scale
  cdef dict dict 

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
    """ converts an internal index to an external index """
    if index & (<node_t>1 << (sizeof(node_t) * 8 - 1)):
      return (index & ~(<node_t>1<<(sizeof(node_t) * 8 - 1))) + self._nodes.used
    else:
      return index

  cdef inline size_t get_length(Tree self) nogil:
    return self._nodes.used + self._leafnodes.used

  cdef inline bint get_node_reclaimable(Tree self, node_t index) nogil:
    return (index != 0) and (self.get_node_parent(index) == -1)

  cdef inline bint get_node_complete(Tree self, node_t index) nogil:
    return (<LeafNode*>self.get_node_pointer(index))[0].complete

  cdef inline void get_node_pos(Tree self, node_t index, double pos[3]) nogil:
    """ returns the topleft corner of the node """
    cdef ipos_t ipos[3]
    fillingcurve.fc2i(self.get_node_key(index), ipos)
    fillingcurve.i2f(self._scale, ipos, pos)

  cdef inline void get_par_pos(Tree self, node_t index, double pos[3]) nogil:
    cdef ipos_t ipos[3]
    fillingcurve.fc2i(self._zkey[self._arg[index]], ipos)
    fillingcurve.i2f(self._scale, ipos, pos)

  cdef inline size_t get_node_npar(Tree self, node_t index) nogil:
    return (<LeafNode*>self.get_node_pointer(index))[0].npar

  cdef inline void set_node_npar(Tree self, node_t index, size_t npar) nogil:
    (<LeafNode*>self.get_node_pointer(index))[0].npar = npar

  cdef inline intptr_t get_node_first(Tree self, node_t index) nogil:
    return (<LeafNode*>self.get_node_pointer(index))[0].first

  cdef inline fckey_t get_node_key(Tree self, node_t index) nogil:
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
    fillingcurve.i2f0(self._scale, isize, size)

  cdef inline node_t get_container(Tree self, double pos[3], int atleast) nogil:
    cdef fckey_t key
    cdef ipos_t ipos[3]
    fillingcurve.f2i(self._scale, pos, ipos)
    fillingcurve.i2fc(ipos, &key)
    return self.get_container_by_key(key, atleast)

  cdef inline node_t get_container_by_key(Tree self, fckey_t key, int atleast) nogil:
    cdef node_t this, child, next
    this = 0
    cdef int nchildren
    cdef node_t * children
    children = self.get_node_children(this, &nchildren)
    while this != -1 and nchildren > 0:
      next = this
      children = self.get_node_children(this, &nchildren)
      for i in range(nchildren):
        if fillingcurve.keyinkey(key, 
             self.get_node_key(children[i]), 
             self.get_node_order(children[i])):
          next = children[i]
          break
      if next == this: # not in any children
        break
      else:
        if self.get_node_npar(next) < atleast: 
          break
        this = next
        continue
    return this

  cdef int _tree_build(Tree self) nogil except -1
  cdef intptr_t _optimize(Tree self) nogil except -1
  cdef node_t _create_child(self, intptr_t first_par, intptr_t parent) nogil except -1
  cdef node_t _try_merge_children(self, intptr_t parent) nogil except -1
  cdef node_t _split_node(self, intptr_t node) nogil except -1

cdef class TreeIter:
  cdef node_t * head[128]
  cdef node_t * end[128]
  cdef readonly Tree tree
  cdef readonly node_t root
  cdef readonly int top

  cdef inline void reset(self, int root) nogil:
    self.root = root
    self.top = -2

  cdef inline node_t get_next_child(self) nogil:
    return self._next(False)
  cdef inline node_t get_next_sibling(self) nogil:
    return self._next(True)
  
  cdef inline node_t _next(self, bint skip_children) nogil:
    """ iterates over all nodes, returns -1 when done,
        if skip_children is non-zero, skip the children and directly go
        siblings  """
    cdef node_t * children
    cdef int nchildren
    # root first, 
    if self.top == -2:
      self.top = -1
      return self.root

    # then first child, if not skip_children
    if self.top == -1:
      if skip_children: return -1
      self.head[0] = self.tree.get_node_children(self.root, &nchildren)
      if nchildren == 0: return -1
      self.end[0] = self.head[0] + nchildren
      self.top = 0
      return self.tree.node_index(self.head[0][0])

    # head[top][0] is the last visited node.
    if not skip_children:
      # avoid a function call of children gonna be skipped anyways
      children = self.tree.get_node_children(self.head[self.top][0], &nchildren)

    if skip_children or nchildren == 0:
      if self.head[self.top] + 1 < self.end[self.top]:
        # visit sibling
        self.head[self.top] = self.head[self.top] + 1
        return self.tree.node_index(self.head[self.top][0])
      else:
        # visit parent sibling
        self.top = self.top - 1
        while self.top >= 0 and self.head[self.top] == self.end[self.top]:
         # keep poping parent till find a sibling
           self.top = self.top - 1
        # no more parents no more siblings, the show is over
        if self.top < 0: return -1
        return self.tree.node_index(self.head[self.top][0])
    else:
      self.head[self.top] = self.head[self.top] + 1
      # push sibling of self.head
      self.top = self.top + 1
      # visit first child on next call
      self.head[self.top] = children
      self.end[self.top] = children + nchildren
      return self.tree.node_index(self.head[self.top][0])

