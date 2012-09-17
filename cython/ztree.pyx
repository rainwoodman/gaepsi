#cython: embedsignature=True
#cython: cdivision=True
import numpy
cimport cpython
from cpython.ref cimport Py_INCREF
cimport numpy
cimport npyiter
cimport npyarray
from libc.stdint cimport *
from libc.stdlib cimport malloc, realloc, free
cdef extern from "math.h":
  int isnan(double d) nogil
cimport cython
import cython
cimport fillingcurve
import fillingcurve
from warnings import warn

DEF BITS = 40

numpy.import_array()

cdef class TreeNode:
  def __cinit__(self, Tree tree, node_t index):
    self.tree = tree
    self._index = index
  property pos:
    """ top-left corner of the node ('f8', 3)"""
    def __get__(self):
      cdef numpy.ndarray pos = numpy.empty(3, dtype='f8')
      self.tree.get_node_pos(self._index, <double*>pos.data)
      return pos
  property size:
    """ size of the node ('f8', 3) """
    def __get__(self):
      cdef numpy.ndarray size = numpy.empty(3, dtype='f8')
      self.tree.get_node_size(self._index, <double*>size.data)
      return size
  property nchildren:
    """ number of children (0 if none) """
    def __get__(self):
      return self.tree.get_node_nchildren(self._index)
  property reclaimable:
    """ if the node is unused """
    def __get__(self):
      return self.tree.get_node_reclaimable(self._index)
  property npar:
    """ number of particles in the node """
    def __get__(self):
      return self.tree.get_node_npar(self._index)
  property first:
    """ index of the first particle """
    def __get__(self):
      return self.tree.get_node_first(self._index)
  property key:
    """ the zkey as a python Long integer"""
    def __get__(self):
      return fillingcurve.fckeyobject(self.tree.get_node_key(self._index))
  property order:
    """ order of the node (bigger = bigger node)"""
    def __get__(self):
      return self.tree.get_node_order(self._index)
  property leafindex:
    """ -1 for a none leaf node, otherwise the index in the leaf node array """
    def __get__(self):
      return self.tree.leafnode_index(self._index)
  property index:
    """ sequentialized node index. smaller indices are the non-leaf nodes, and bigger ones are the leaf nodes """
    def __get__(self):
      return self.tree.node_index(self._index)
  property parent:
    """ index of the parent, must be a non leaf node """
    def __get__(self):
      return self.tree.get_node_parent(self._index)
  property children:
    """ sequential indices of childrens as an 'intp' array. The internal array(where leaf nodes are prefixed with highest bit set to 1) is NOT returned """
    def __get__(self):
      cdef int nchildren, k
      children = self.tree.get_node_children(self._index, &nchildren)
      cdef numpy.ndarray ret = numpy.empty(nchildren, dtype='intp')
      for k in range(nchildren):
        (<intptr_t *>ret.data)[k] = self.tree.node_index(children[k])
      return ret

  def contains(self, points, out=None):
    """ test if points are inside a node.
        accept positions,
        Python Long integer arrays as zkeys,
        or arrays as zkeys(returned by the digitizer)
    """
    
    points = numpy.asarray(points)
    cdef object dtype
    if points.dtype in (fillingcurve.fckeytype, numpy.dtype('object')):
      X, Y, Z = points, None, None
      return fillingcurve.contains(self.key, self.order, X, Y, Z, out=out)
    elif len(points.shape) > 1 and points.shape[-1] == 3:
      X, Y, Z = points.rollaxis(-1)
      if X.dtype.kind == 'i':
        return fillingcurve.contains(self.key, self.order, X, Y, Z, out=out, scale=None)
      elif X.dtype.kind == 'f':
        return fillingcurve.contains(self.key, self.order, X, Y, Z, out=out, scale=self.tree.scale)
      else:
        raise RuntimeError('dtype not supported')
    else:
      raise RuntimeError('dtype not supported %s' % str(points.dtype))

  def __str__(self):
    return str(dict(parent=self.parent, children=self.children,
               index=self.index, leafindex=self.leafindex,
               order=self.order, key=self.key,
               first=self.first, npar=self.npar,
               pos=self.pos, size=self.size))
  def __repr__(self):
    return "TreeNode(%s, %s): %s" % (repr(self.tree), repr(self._index), str(self))

cdef class TreeProperty:
  """ Property associated with tree nodes/particles """
  cdef readonly Tree tree
  cdef readonly numpy.ndarray values
  cdef readonly numpy.ndarray cache
  cdef float * _cache
  cdef bint direct
  cdef npyarray.CArray cvalues
  def __cinit__(self):
    pass

  def __init__(self, tree, values):
    self.tree = tree
    if isinstance(values, basestring):
      self.cache = self._project(values)
      self.direct = True
    else:
      self.cache = numpy.empty(len(tree), 'f4')
      self.cache[:] = numpy.nan
      self._cache = <float *> self.cache.data
      self.direct = False
      self._prepare(self, values)

  def _prepare(self, numpy.ndarray nodes, values):
    self._cache = <float *> nodes.data
    self.values = numpy.asarray(values)
    npyarray.init(&self.cvalues, values)

  def __len__(self):
    return len(self.tree)

  def __getitem__(self, item):
    if self.direct: return self.cache[item]
    if numpy.isscalar(item):
      return self.get_node_value(item)
    else:
      nanmask = numpy.isnan(self.cache[item])
      if nanmask.any():
        nanlist = numpy.arange(len(self.cache))[item][nanmask]
        for k in nanlist:
          self.get_node_value(k)
    return self.cache[item]

  cdef numpy.ndarray _project(self, prop):
    cdef intptr_t i
    cdef intptr_t method 
    actions = {
       'key':(fillingcurve.fckeytype, <intptr_t> self.tree.get_node_key),
       'used':('?', <intptr_t> -1),
       'order':('i2', <intptr_t> self.tree.get_node_order),
       'first':('intp', <intptr_t> self.tree.get_node_first),
       'npar':('intp', <intptr_t> self.tree.get_node_npar),
       'parent':('intp', <intptr_t> self.tree.get_node_parent),
       'nchildren':('i2', <intptr_t> self.tree.get_node_nchildren),
       'pos':(('f8', 3), <intptr_t> self.tree.get_node_pos),
       'size':(('f8', 3), <intptr_t> self.tree.get_node_size)
    }

    dtype, method = actions[prop]
    
    out = numpy.empty(shape=len(self), dtype=dtype)
    if isinstance(dtype, tuple):
     ops = [out[..., d] for d in range(dtype[1])]
     op_flags = [['writeonly']] * dtype[1]
     op_dtypes = [dtype[0]] * dtype[1]
    else:
     ops = [out]
     op_flags = [['writeonly']]
     op_dtypes = [dtype]

    iter = numpy.nditer(
        ops,
        op_flags=op_flags,
        flags = ['buffered', 'external_loop', 'refs_ok'],
        casting='unsafe',
        op_dtypes=op_dtypes)
      
    cdef npyiter.CIter citer
    cdef size_t size = npyiter.init(&citer, iter)
    cdef double fdata[3]
    cdef fckey_t key
    i = 0
    while size > 0:
      while size > 0:
        if method == <intptr_t> -1:
          (<char*>(citer.data[0]))[0] = ~self.tree.get_node_reclaimable(i)
        elif method == <intptr_t>self.tree.get_node_key:
          (<fckey_t *>(citer.data[0]))[0] = self.tree.get_node_key(i)
        elif method == <intptr_t>self.tree.get_node_first:
          (<intptr_t *>(citer.data[0]))[0] = self.tree.get_node_first(i)
        elif method == <intptr_t>self.tree.get_node_parent:
          (<intptr_t *>(citer.data[0]))[0] = self.tree.get_node_parent(i)
        elif method == <intptr_t>self.tree.get_node_npar:
          (<intptr_t *>(citer.data[0]))[0] = self.tree.get_node_npar(i)
        elif method == <intptr_t>self.tree.get_node_nchildren:
          (<short int *>(citer.data[0]))[0] = self.tree.get_node_nchildren(i)
        elif method == <intptr_t>self.tree.get_node_order:
          (<short int *>(citer.data[0]))[0] = self.tree.get_node_order(i)
        elif method == <intptr_t>self.tree.get_node_pos:
          self.tree.get_node_pos(i, fdata)
          (<double *>(citer.data[0]))[0] = fdata[0]
          (<double *>(citer.data[1]))[0] = fdata[1]
          (<double *>(citer.data[2]))[0] = fdata[2]
        elif method == <intptr_t>self.tree.get_node_size:
          self.tree.get_node_size(i, fdata)
          (<double *>(citer.data[0]))[0] = fdata[0]
          (<double *>(citer.data[1]))[0] = fdata[1]
          (<double *>(citer.data[2]))[0] = fdata[2]
        else:
          raise IndexError("%s" % prop)
        npyiter.advance(&citer)
        size = size - 1
        i = i + 1
      size = npyiter.next(&citer) 
    return out

  cdef float get_node_value(self, node_t node) nogil:
    cdef int nchildren
    cdef int k
    cdef float value, abit
    cdef intptr_t first
    node = self.tree.node_index(node)
    value = (<float*>self._cache)[node]
    if isnan(value):
      value = 0
      children = self.tree.get_node_children(node, &nchildren)
      if nchildren == 0:
        first = self.tree.get_node_first(node)
        for k in range(self.tree.get_node_npar(node)):
          npyarray.flat(&self.cvalues, k+first, &abit)
          value += abit
      else:
        for k in range(nchildren):
          value += self.get_node_value(children[k])
      (<float*>self._cache)[node] = value
    return value

cdef class Tree:
  
  def __cinit__(self):
    flexarray.init(&self._nodes, <void**>&self.nodes, sizeof(Node), 1024)
    flexarray.init(&self._leafnodes, <void**>&self.leafnodes, sizeof(LeafNode), 1024)

  def __init__(self, zkey, scale, thresh=100):
    """ scale is min[0], min[1], min[2], norm( multply by norm to go to 0->1<<BITS -1)
        zkey needs to be sorted!"""
    self.maxthresh = thresh
    self.minthresh = 1
    self.propertyvalues = {}
    self.scale = numpy.empty(4, dtype='f8')
    self._scale = <double*> self.scale.data
    self._scale[0] = scale[0]
    self._scale[1] = scale[1]
    self._scale[2] = scale[2]
    self._scale[3] = scale[3]
    if zkey.dtype != fillingcurve.fckeytype:
      raise TypeError("zkey needs to be of %s" % str(fillingcurve.fckeytype))
    self.zkey = zkey
    self._zkey = <fckey_t *> self.zkey.data
    self._zkey_length = self.zkey.shape[0]
    if -1 == self._tree_build():
     pass
     # raise ValueError("tree build failed. Is the input zkey sorted?")

  def optimize(self):
    while self._optimize() > 0: continue

  def split_tail(self):
    self._split_node(mark_leaf(self._leafnodes.used - 1))

  def transverse(self, prefunc=None, postfunc=None, index=0):
    node = self[index]
    if prefunc:    prefunc(node)
    if len(node['children']) > 0:
      for i in node['children']:
        self.transverse(prefunc=prefunc, postfunc=postfunc, index=i)
    if postfunc: postfunc(node)

  def __iter__(self):
    def func():
      cdef node_t j = 0
      for j in range(self.node_length + self.leaf_length):
        node = self[j]
        yield node
    return func()

  def __setitem__(self, item, value):
    self.propertyvalues[item] = value

  def __getitem__(self, item):
    if isinstance(item, basestring):
      if item in self.propertyvalues:
        return TreeProperty(self, self.propertyvalues[item])
      else:
        return TreeProperty(self, item)
    elif isinstance(item, slice):
        start, stop, step = slice.indices(len(self))
        return [TreeNode(self, i) for i in range(start, stop, step)]
    elif isinstance(item, numpy.ndarray) and item.dtype == numpy.dtype('?'):
      return [TreeNode(self, i) for i in range(len(self)) if item[i]]
    elif hasattr(item, '__iter__'):
      return [TreeNode(self, i) for i in item]
    elif numpy.isscalar(item):
      return TreeNode(self, item)
    else:
      raise IndexError(str(item))

  cdef int _tree_build(Tree self) nogil except -1:
      cdef intptr_t j = 0
      cdef intptr_t i = 0
      cdef intptr_t extrastep = 0

      flexarray.append(&self._nodes, 1)
      while i < self._zkey_length and self._zkey[i] == -1: 
        i = i + 1
      self.nodes[0].key = 0
      self.nodes[0].first = i
      self.nodes[0].npar = 0
      self.nodes[0].order = BITS
      self.nodes[0].parent = -1
      self.nodes[0].child_length = 0
      while i < self._zkey_length:
        while not j == -1 and \
          not fillingcurve.keyinkey(self._zkey[i], self.get_node_key(j), self.get_node_order(j)):
          # close the nodes by filling in the npar, because we already scanned over
          # all particles in these nodes.
          self.set_node_npar(j, i - self.get_node_first(j))

          # if the immediate parent is not full, merge it.
          # this will save hack a lot of memory by performing
          # the merge and free up the nodes at the end of the queue
          # instead of digging up holes in the end with 'optimize'.
          # however we still need to run 'optimize' in the end
          # because this won't catch all cases.
          j = self._try_merge_children(j)
          j = self.get_node_parent(j)
          # because we are on a morton key ordered list, no need to deccent into children 
        if j == -1: # par not covered by the tree anywhere, skip it.
          with gil:
            raise RuntimeError('remature ending at %d / %d' (i, self._zkey_length))
        # NOTE: will never go beyond 8 children per node, 
        # for the child_length > 0 branch is called less than 8 times on the parent, 

        # ASSERTION: 
        # 1) the point(i) is in the current node(j), 
        # 2) not in any of the current children of node(j) [guarrenteed by morton key sorting]
        if self.get_node_nchildren(j) > 0:
          # already not a leaf, create new child
          j = self._create_child(i, j) 
        elif (self.get_node_npar(j) >= self.minthresh and self.get_node_order(j) > 0):
          # too many points in the leaf, split it
          # NOTE: i is rewinded, because now some of the particles are no longer
          # in the new node.
          i = self.get_node_first(j)
          j = self._create_child(i, j) 
        else:
          # put the particle into the leaf.
          pass
        # if error occured
        if j == -1: 
          with gil: raise RuntimeError('parent node is -1 at i = %d' % i)
        # particle i is not in node j yet.
        # next particle to be considered is i+1,
        # now we try to fast forword to the first particle that is not in the current node
        if self.minthresh > 1:
          extrastep = self.minthresh - self.get_node_npar(j) - 1
          if extrastep > 0 and i + extrastep < self._zkey_length:
            while not fillingcurve.keyinkey(self._zkey[i+extrastep], self.get_node_key(j), self.get_node_order(j)):
              extrastep >>= 1
          else:
            extrastep = 0
        else:
          extrastep = 0
        self.set_node_npar(j, self.get_node_npar(j) + extrastep + 1)
        i = i + 1 + extrastep
      # now close the remaining open nodes
      while j != -1:
        self.set_node_npar(j, i - self.get_node_first(j))
        j = self._try_merge_children(j)
        j = self.get_node_parent(j)

  cdef node_t _split_node(self, intptr_t node) nogil except -1:
    cdef intptr_t i, end
    cdef size_t npar
    cdef node_t j
    cdef node_t grandparent = self.get_node_parent(node)
    i = self.get_node_first(node)
    npar = self.get_node_npar(node)
    end = i + npar
    j = self._create_child(i, node)
    i = i + 1
    while i < end:
        while not j == grandparent and \
          not fillingcurve.keyinkey(self._zkey[i], self.get_node_key(j), self.get_node_order(j)):
          self.set_node_npar(j, i - self.get_node_first(j))
          j = self.get_node_parent(j)
        if j == grandparent: 
          #with gil: raise RuntimeError('reaching grand parent')
          pass

        if self.get_node_nchildren(j) > 0:
          # already not a leaf, create new child
          j = self._create_child(i, j) 
        else:
          pass
        # if error occured
        if j == -1: 
          #with gil: raise RuntimeError('parent node is -1 at i = %d' % i)
          pass
        if self.get_node_order(j) == 0:
          #with gil: raise RuntimeError("trying to split a order 0 node. can't resolve this")
          pass

        self.set_node_npar(j, self.get_node_npar(j) + 1)
        i = i + 1
      # now close the remaining open nodes
    while j != grandparent:
      self.set_node_npar(j, i - self.get_node_first(j))
      j = self.get_node_parent(j)
    return j

  cdef node_t _create_child(self, intptr_t first_par, intptr_t parent) nogil except -1:
    # notice that parent node will be nolonger a valid node, because it has been
    # replaced by a non-leaf node.
    cdef intptr_t fullparent 
    cdef node_t index
    cdef int k
    if self.is_leafnode(parent):
      index = self.leafnode_index(parent)
      # we need to move the parent from leafnode to the node storage pool
      fullparent = flexarray.append(&self._nodes, 1)
      # first copy
      self.nodes[fullparent].key = self.leafnodes[index].key
      self.nodes[fullparent].first = self.leafnodes[index].first
      self.nodes[fullparent].order = self.leafnodes[index].order
      self.nodes[fullparent].npar = self.leafnodes[index].npar
      self.nodes[fullparent].parent = self.leafnodes[index].parent
      self.nodes[fullparent].child_length = 0
      # now replace in the grandparent's children.
      grandparent = self.nodes[fullparent].parent
      for k in range(8):
        if self.nodes[grandparent].child[k] == parent:
          self.nodes[grandparent].child[k] = fullparent
          break
      # the following test shall always be fine because we
      # only do try split the last leaf node with add_child
      if index != self._leafnodes.used - 1:
        #with gil: raise RuntimeError('not called on the last node')
        pass
      parent = fullparent
      flexarray.remove(&self._leafnodes, 1)

    index = flexarray.append(&self._leafnodes, 1)
    # creates a child of parent from first_par, returns the new child */
    self.leafnodes[index].first = first_par
    self.leafnodes[index].npar = 0
    self.leafnodes[index].order = self.nodes[parent].order - 1
    # the lower bits of a sqkey is cleared off, so that get_node_pos returns 
    # correct corner coordinates
    self.leafnodes[index].key = fillingcurve.truncate(self._zkey[first_par], self.leafnodes[index].order)
    self.leafnodes[index].parent = parent
    self.leafnodes[index].child_length = 0

    # the follwing assertion shall never be true unless the input
    # keys are not properly sorted.
    if self.nodes[parent].child_length >= 8:
      #with gil: raise RuntimeError("child_length >= 8,  parent = %d %s %d %s" % (parent, str(self[parent]), first_par, str(self.zkey[first_par])))
      pass

    index = mark_leaf(index)
    self.nodes[parent].child[self.nodes[parent].child_length] = index
    self.nodes[parent].child_length = self.nodes[parent].child_length + 1

    return index 

  cdef node_t _try_merge_children(Tree self, intptr_t parent) nogil:
    cdef int nchildren
    cdef int k
    cdef node_t index, grandparent
    cdef node_t * children

    # do not merge if the node is already a leaf node
    if self.leafnode_index(parent) != -1: return parent

#    if self.get_node_nchildren(parent) >= 7:
#      return parent

    if self.get_node_npar(parent) > self.maxthresh: 
      return parent

    grandparent = self.nodes[parent].parent
    if grandparent == -1: return parent

    nchildren = self.nodes[parent].child_length

    children = self.nodes[parent].child

    # see if this is an immediate parent of leaf nodes
    for k in range(nchildren):
      index = self.leafnode_index(children[k])
      if index == -1: return parent

    # sanity check, shall disable later
    # if the leaf nodes are discontinues, give up
    # do not merge two loops(this and the above)
    # it can happen that a node is rejected by the first
    # for later elements, yet the second fails.
    for k in range(nchildren):
      index = self.leafnode_index(children[k])
      # the following shall never happen because we only call try_merge
      # on the most recently created node.
      #
      if index + nchildren - k != self._leafnodes.used:
         return parent
      #with gil: print '%d != %d' % (index + nchildren - k , self._leafnodes.used)

    if True: # for indentation
      # first mark the children reclaimable
      for k in range(nchildren):
        index = self.leafnode_index(children[k])
#        if index == -1: continue  # this is never true
        self.leafnodes[index].parent = -1

      index = self.leafnode_index(children[0])
      # index points to a good leafnode, b/c it has been merged
      # then copy the parent node to a leafnode(replace the first children)
      self.leafnodes[index].key = self.nodes[parent].key
      self.leafnodes[index].first = self.nodes[parent].first
      self.leafnodes[index].order = self.nodes[parent].order
      self.leafnodes[index].npar = self.nodes[parent].npar
      self.leafnodes[index].parent = grandparent

      # update the grandparent
      for k in range(self.nodes[grandparent].child_length):
        if self.nodes[grandparent].child[k] == parent:
          self.nodes[grandparent].child[k] = mark_leaf(index)

      # mark old parent node reclaimable
      self.nodes[parent].npar = 0
      self.nodes[parent].parent = -1

      # free up the actual memory.
      # notice that when try_merge_children is called,
      # parent is always the last item in _nodes,
      # and its children are always the last children in _leafnodes.

      flexarray.remove(&self._leafnodes, nchildren - 1)
      flexarray.remove(&self._nodes, 1)
    return mark_leaf(index)

  cdef intptr_t _optimize(Tree self) nogil except -1:
    # merge the immediate parent of leaf nodes if it is incomplete.
    cdef intptr_t j
    cdef intptr_t changed = 0
    cdef node_t parent
    for j in range(self._leafnodes.used):
      # skip the reclaimable ones
      parent = self.leafnodes[j].parent
      if parent == -1: continue
      if parent != self._try_merge_children(parent):
        change = changed + 1
    return changed

  def __len__(self):
    return self.get_length()

  property node_length:
    def __get__(self):
      return self._nodes.used
    
  property leaf_length:
    def __get__(self):
      return self._leafnodes.used

  def __dealloc__(self):
    flexarray.destroy(&self._nodes)
    flexarray.destroy(&self._leafnodes)


