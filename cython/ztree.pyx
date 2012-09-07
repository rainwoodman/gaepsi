#cython: embedsignature=True
#cython: cdivision=True
import numpy
cimport cpython
from cpython.ref cimport Py_INCREF
cimport numpy
cimport npyiter
from libc.stdint cimport *
from libc.stdlib cimport malloc, realloc, free
cimport cython
import cython
from warnings import warn
cimport zorder
from zorder cimport zorder_t, _zorder_dtype

numpy.import_array()

cdef class Tree:

  def __cinit__(self):
    flexarray.init(&self._nodes, <void**>&self.nodes, sizeof(Node), 1024)
    flexarray.init(&self._leafnodes, <void**>&self.leafnodes, sizeof(LeafNode), 1024)

  def __init__(self, zkey, digitize, thresh=100):
    """ digitize is an zorder.Digitize object.
        zkey needs to be sorted!"""
    self.thresh = thresh
    self.digitize = digitize
    if not zkey.dtype == _zorder_dtype:
      raise TypeError("zkey needs to be of %s" % _zorder_dtype.str)
    self.zkey = zkey
    self._zkey = <zorder_t *> self.zkey.data
    self._zkey_length = self.zkey.shape[0]
    if -1 == self._tree_build():
      raise ValueError("tree build failed. Is the input zkey sorted?")

  def optimize(self):
    while self._optimize() > 0: continue

  def transverse(self, prefunc=None, postfunc=None, index=0):
    node = self[index]
    if prefunc:    prefunc(node)
    if len(node['children']) > 0:
      for i in node['children']:
        self.transverse(prefunc=prefunc, postfunc=postfunc, index=i)
    if postfunc: postfunc(node)
    
  def getnode(self, ind):
    """ returns a dictionary of the tree node by ind,
        or if ind is 
           'npar', 'first', 'last', 'pos', 'size', 'parent'
        return array
    """
    cdef numpy.intp_t dims[1]
    cdef int nchildren
    cdef node_t * children
    cdef numpy.ndarray arr
    cdef int k

    children = self.get_node_children(ind, &nchildren)
    dims[0] = nchildren
    #FIXME: watchout!! NPY_INT
    arr = numpy.PyArray_SimpleNew(1, dims, numpy.NPY_INT)
    for k in range(nchildren):
      (<int*>(arr.data))[k] = self.node_index(children[k])

    cdef numpy.ndarray pos
    cdef numpy.ndarray size
    pos = numpy.empty(3, dtype='f8')
    size = numpy.empty(3, dtype='f8')
    self.get_node_pos(ind, <double*>pos.data)
    self.get_node_size(ind, <double*>size.data)


    rt = dict(key=self.get_node_key(ind),
            order=self.get_node_order(ind),
           parent=self.get_node_parent(ind),
            index=ind,
        nchildren=nchildren,
        leafindex=self.leafnode_index(ind),
            first=self.get_node_first(ind), 
             last=self.get_node_npar(ind) + self.get_node_first(ind),
             npar=self.get_node_npar(ind),
              pos=pos, 
             size=size,
         children=arr)

    return rt
    
  def getprop(self, prop):
    cdef intptr_t i
    cdef intptr_t method 
    actions = {
       'key':('object', <intptr_t> self.get_node_key),
       'mask':('?', <intptr_t> -1),
       'order':('i2', <intptr_t> self.get_node_order),
       'first':('intp', <intptr_t> self.get_node_first),
       'npar':('intp', <intptr_t> self.get_node_npar),
       'parent':('intp', <intptr_t> self.get_node_parent),
       'nchildren':('i2', <intptr_t> self.get_node_nchildren),
       'pos':(('f8', 3), <intptr_t> self.get_node_pos),
       'size':(('f8', 3), <intptr_t> self.get_node_size)
    }

    dtype, method = actions[prop]
    
    out = numpy.empty(len(self), dtype=dtype)
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
    cdef object obj
    i = 0
    while size > 0:
      while size > 0:
        if method == <intptr_t> -1:
          (<bint *>(citer.data[0]))[0] = (self.get_node_first(i) != -1)
        if method == <intptr_t>self.get_node_key:
          obj = self.get_node_key(i)
          (<void**>(citer.data[0]))[0] = <void*>obj
          cpython.Py_INCREF(obj)
        elif method == <intptr_t>self.get_node_first:
          (<intptr_t *>(citer.data[0]))[0] = self.get_node_first(i)
        elif method == <intptr_t>self.get_node_parent:
          (<intptr_t *>(citer.data[0]))[0] = self.get_node_parent(i)
        elif method == <intptr_t>self.get_node_npar:
          (<intptr_t *>(citer.data[0]))[0] = self.get_node_npar(i)
        elif method == <intptr_t>self.get_node_nchildren:
          (<short int *>(citer.data[0]))[0] = self.get_node_nchildren(i)
        elif method == <intptr_t>self.get_node_order:
          (<short int *>(citer.data[0]))[0] = self.get_node_order(i)
        elif method == <intptr_t>self.get_node_pos:
          self.get_node_pos(i, fdata)
          (<double *>(citer.data[0]))[0] = fdata[0]
          (<double *>(citer.data[1]))[0] = fdata[1]
          (<double *>(citer.data[2]))[0] = fdata[2]
        elif method == <intptr_t>self.get_node_size:
          self.get_node_size(i, fdata)
          (<double *>(citer.data[0]))[0] = fdata[0]
          (<double *>(citer.data[1]))[0] = fdata[1]
          (<double *>(citer.data[2]))[0] = fdata[2]

        npyiter.advance(&citer)
        size = size - 1
        i = i + 1
      size = npyiter.next(&citer) 
    return out
  def __getitem__(self, item):
    if isinstance(item, basestring):
      return self.getprop(item)
    elif isinstance(item, slice):
      start, stop, step = slice.indices(len(self))
      return [self.getnode(i) for i in range(start, stop, step)]
    else:
      return self.getnode(item)
    
  cdef int _tree_build(Tree self) nogil:
      cdef intptr_t j = 0
      cdef intptr_t i = 0
      cdef intptr_t extrastep = 0
      flexarray.append(&self._nodes, 1)
      self.nodes[0].key = 0
      self.nodes[0].first = 0
      self.nodes[0].npar = 0
      self.nodes[0].order = self.digitize.bits
      self.nodes[0].parent = -1
      self.nodes[0].child_length = 0
      while i < self._zkey_length:
        while not zorder.boxtest(self.get_node_key(j), self.get_node_order(j), self._zkey[i]):
          # close the nodes by filling in the npar, because we already scanned over
          # all particles in these nodes.
          self.set_node_npar(j, i - self.get_node_first(j))
          j = self.get_node_parent(j)
          # because we are on a morton key ordered list, no need to deccent into children 
        # NOTE: will never go beyond 8 children per node, 
        # for the child_length > 0 branch is called less than 8 times on the parent, 

        # ASSERTION: the point(i) fails boxtest of the current node(j)
        if self.get_node_nchildren(j) > 0:
          # already not a leaf, create new child
          j = self._create_child(i, j) 
        elif (self.get_node_npar(j) >= self.thresh and self.get_node_order(j) > 0):
          # too many points in the leaf, split it
          # NOTE: i is rewinded, because now some of the particles are no longer
          # in the new node.
          i = self.get_node_first(j)
          j = self._create_child(i, j) 
        else:
          # put the particle into the leaf.
          pass
        # particle i is not in node j yet.
        # next particle to be considered is i+1,
        if j == -1: 
          return -1
        # now we try to fast forword to the first particle that is not in the current node
        extrastep = self.thresh - self.get_node_npar(j) - 1
        if extrastep > 0 and i + extrastep < self._zkey_length:
          while not zorder.boxtest(self.get_node_key(j), self.get_node_order(j), self._zkey[i + extrastep]):
            extrastep >>= 1
        else:
          extrastep = 0
        self.set_node_npar(j, self.get_node_npar(j) + extrastep + 1)
        i = i + 1 + extrastep
      # now close the remaining open nodes
      while j != -1:
        self.set_node_npar(j, i - self.get_node_first(j))
        j = self.get_node_parent(j)
        
  cdef node_t _create_child(self, intptr_t first_par, intptr_t parent) nogil:
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
      if index != self._leafnodes.used - 1:
        with gil:
          raise Exception('consistency %d != %d' %(parent, self._leafnodes.used - 1))
      parent = fullparent
      flexarray.remove(&self._leafnodes, 1)

    index = flexarray.append(&self._leafnodes, 1)
    # creates a child of parent from first_par, returns the new child */
    self.leafnodes[index].first = first_par
    self.leafnodes[index].npar = 0
    self.leafnodes[index].order = self.nodes[parent].order - 1
    # the lower bits of a sqkey is cleared off, so that get_node_pos returns 
    # correct corner coordinates
    self.leafnodes[index].key = zorder.truncate(self._zkey[first_par], self.leafnodes[index].order)
    self.leafnodes[index].parent = parent

    index = index | (<int> 1 << 31)
    self.nodes[parent].child[self.nodes[parent].child_length] = index
    self.nodes[parent].child_length = self.nodes[parent].child_length + 1

    if self.nodes[parent].child_length > 8:
      with gil: raise RuntimeError("child_length > 8")
      return -1

    return index 

  cdef intptr_t _optimize(Tree self) nogil:
    # merge the immediate parent of leaf nodes if it is incomplete.
    cdef intptr_t j
    cdef intptr_t changed = 0
    cdef node_t parent
    cdef node_t * children
    cdef int nchildren
    cdef node_t grandparent
    cdef intptr_t index
    cdef node_t freechild
    cdef int notimmediate

    for j in range(self._leafnodes.used):
      if self.leafnodes[j].first == -1: continue
      parent = self.leafnodes[j].parent
      nchildren = self.nodes[parent].child_length
      grandparent = self.nodes[parent].parent
      if grandparent == -1:
        if parent != 0:
          with gil:
            raise RuntimeError("visiting a parent twice %d" % parent)
        continue

      if nchildren == 8: continue
        
      children = self.nodes[parent].child
      notimmediate = 0
      for k in range(nchildren):
        index = self.leafnode_index(children[k])
        if index == -1: 
          notimmediate = 1
          break
         
      # only do nodes with all leaf type children
      if notimmediate: continue

      changed = changed + 1
      # first mark the children reclaimable
      for k in range(nchildren):
        index = self.leafnode_index(children[k])
        if index == -1: continue
        self.leafnodes[index].first = -1
        self.leafnodes[index].parent = -1
        self.leafnodes[index].npar = 0

      # j points to a good leafnode, b/c it has been merged
      # then copy the parent node to a leafnode(replace the first children)
      self.leafnodes[j].key = self.nodes[parent].key
      self.leafnodes[j].first = self.nodes[parent].first
      self.leafnodes[j].order = self.nodes[parent].order
      self.leafnodes[j].npar = self.nodes[parent].npar
      self.leafnodes[j].parent = grandparent

      # update the grandparent
      for k in range(self.nodes[grandparent].child_length):
        if self.nodes[grandparent].child[k] == parent:
          self.nodes[grandparent].child[k] = j | (<int>1 << 31)

      # mark old parent node reclaimable
      self.nodes[parent].first = -1
      self.nodes[parent].npar = 0
      self.nodes[parent].parent = -1
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


