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

  def transverse(self, prefunc=None, postfunc=None, index=0):
    node = self[index]
    if prefunc:    prefunc(node)
    if len(node['children']) > 0:
      for i in node['children']:
        self.transverse(prefunc=prefunc, postfunc=postfunc, index=i)
    if postfunc: postfunc(node)
    
  def __getitem__(self, ind):
    """ returns a dictionary of the tree node by ind """
    cdef numpy.intp_t dims[1]
    cdef int nchildren
    cdef node_t * children
    children = self.get_node_children(ind, &nchildren)
    dims[0] = nchildren
    arr = numpy.PyArray_SimpleNewFromData(1, dims, numpy.NPY_INT, children)
    numpy.set_array_base(arr, self)
    rt = dict(key=self.get_node_key(ind),
            order=self.get_node_order(ind),
           parent=self.get_node_parent(ind),
            index=ind,
         children=arr)

    cdef numpy.ndarray pos
    cdef numpy.ndarray size
    pos = numpy.empty(3, dtype='f8')
    size = numpy.empty(3, dtype='f8')
    self.get_node_pos(ind, <double*>pos.data)
    self.get_node_size(ind, <double*>size.data)

    rt.update(dict(pos=pos, size=size))

    rt.update(dict(first=self.get_node_first(ind), 
                      last=self.get_node_npar(ind) + self.get_node_first(ind)
             ))
    return rt
    

  cdef int _tree_build(Tree self) nogil:
      cdef intptr_t j = 0
      cdef intptr_t i = 0
      cdef intptr_t step = 0
      flexarray.append(&self._nodes, 1)
      self.nodes[0].key = 0
      self.nodes[0].first = 0
      self.nodes[0].npar = 0
      self.nodes[0].order = self.digitize.bits
      self.nodes[0].parent = -1
      self.nodes[0].child_length = 0
      while i < self._zkey_length:
        while not zorder.boxtest(self.nodes[j].key, self.nodes[j].order, self._zkey[i]):
          # close the nodes by filling in the npar, because we already scanned over
          # all particles in these nodes.
          self.nodes[j].npar = i - self.nodes[j].first
          j = self.nodes[j].parent
          # because we are on a morton key ordered list, no need to deccent into children 
        # NOTE: will never go beyond 8 children per node, 
        # for the child_length > 0 branch is called less than 8 times on the parent, 
        # the point fails in_square of the current node
        if self.nodes[j].child_length > 0:
          # already not a leaf, create new child
          j = self._create_child(i, j) 
        elif (self.nodes[j].npar > self.thresh and self.nodes[j].order > 0):
          # too many points in the leaf, split it
          # NOTE: i is rewinded, because now some of the particles are no longer
          # in the new node.
          i = self.nodes[j].first
          j = self._create_child(i, j) 
        else:
          # put the particle into the leaf.
          self.nodes[j].npar = self.nodes[j].npar + 1
        if j == -1: return -1
        # now we try to fast forword to the first particle that is not in the current node
        step = self.thresh
        if i + step < self._zkey_length:
          while not zorder.boxtest(self.nodes[j].key, self.nodes[j].order, self._zkey[i + step]):
            step >>= 1
          if step > 0:
            self.nodes[j].npar = self.nodes[j].npar + step
            i = i + step
        i = i + 1
      # now close the remaining open nodes
      while j >= 0:
        self.nodes[j].npar = i - self.nodes[j].first
        j = self.nodes[j].parent
        
        
  cdef node_t _create_child(self, intptr_t first_par, intptr_t parent) nogil:
    cdef intptr_t index = flexarray.append(&self._nodes, 1)
    # creates a child of parent from first_par, returns the new child */
    self.nodes[index].first = first_par
    self.nodes[index].npar = 1
    self.nodes[index].parent = parent
    self.nodes[index].child_length = 0
    self.nodes[index].order = self.nodes[parent].order - 1
    # the lower bits of a sqkey is cleared off, so that get_node_pos returns 
    # correct corner coordinates
    self.nodes[index].key = zorder.truncate(self._zkey[first_par], self.nodes[index].order)
    self.nodes[parent].child[self.nodes[parent].child_length] = index
    self.nodes[parent].child_length = self.nodes[parent].child_length + 1

    if self.nodes[parent].child_length > 8:
      return -1
    return index

  def __dealloc__(self):
    flexarray.destroy(&self._nodes)


