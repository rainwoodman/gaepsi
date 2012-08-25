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

NodeInfoDtype = numpy.dtype([('key', _zorder_dtype), ('order', 'i2'), ('child_length', 'i2'), ('parent', 'i4'), ('first', 'i8'), ('npar', 'i8'), ('child', ('i4', 8))])

cdef class Tree:

  def __cinit__(self):
    self.size = 1024
    self._nodes = <NodeInfo *>malloc(sizeof(NodeInfo) * self.size)
    self.used = 0

  @cython.boundscheck(False)
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

  property nodes:
    def __get__(self):
      """ returns the internal buffer as a recarray, not very useful"""
      cdef numpy.intp_t dims[1]
      dims[0] = self.used * sizeof(NodeInfo)
      arr = numpy.PyArray_SimpleNewFromData(1, dims, numpy.NPY_BYTE, self._nodes)
      numpy.set_array_base(arr, self)
      return arr.view(dtype=NodeInfoDtype)

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
    dims[0] = self._nodes[ind].child_length
    arr = numpy.PyArray_SimpleNewFromData(1, dims, numpy.NPY_INT, self._nodes[ind].child)
    numpy.set_array_base(arr, self)
    rt = dict(key=self._nodes[ind].key, 
            order=self._nodes[ind].order,
           parent=self._nodes[ind].parent, 
            index=ind,
         children=arr)

    cdef numpy.ndarray pos
    cdef numpy.ndarray size
    pos = numpy.empty(3, dtype='f8')
    size = numpy.empty(3, dtype='f8')
    self.get_node_pos(ind, <double*>pos.data)
    self.get_node_size(ind, <double*>size.data)

    rt.update(dict(pos=pos, size=size))

    rt.update(dict(first=self._nodes[ind].first, 
                      last=self._nodes[ind].first + self._nodes[ind].npar))
    return rt
    

  @cython.boundscheck(False)
  cdef int _tree_build(Tree self) nogil:
      cdef intptr_t j = 0
      cdef intptr_t i = 0
      cdef intptr_t step = 0
      self.used = 1;
      self._nodes[0].key = 0
      self._nodes[0].first = 0
      self._nodes[0].npar = 0
      self._nodes[0].order = self.digitize.bits
      self._nodes[0].parent = -1
      self._nodes[0].child_length = 0
      while i < self._zkey_length:
        while not zorder.boxtest(self._nodes[j].key, self._nodes[j].order, self._zkey[i]):
          # close the nodes by filling in the npar, because we already scanned over
          # all particles in these nodes.
          self._nodes[j].npar = i - self._nodes[j].first
          j = self._nodes[j].parent
          # because we are on a morton key ordered list, no need to deccent into children 
        # NOTE: will never go beyond 8 children per node, 
        # for the child_length > 0 branch is called less than 8 times on the parent, 
        # the point fails in_square of the current node
        if self._nodes[j].child_length > 0:
          # already not a leaf, create new child
          j = self._create_child(i, j) 
        elif (self._nodes[j].npar > self.thresh and self._nodes[j].order > 0):
          # too many points in the leaf, split it
          # NOTE: i is rewinded, because now some of the particles are no longer
          # in the new node.
          i = self._nodes[j].first
          j = self._create_child(i, j) 
        else:
          # put the particle into the leaf.
          self._nodes[j].npar = self._nodes[j].npar + 1
        if j == -1: return -1
        # now we try to fast forword to the first particle that is not in the current node
        step = self.thresh
        if i + step < self._zkey_length:
          while not zorder.boxtest(self._nodes[j].key, self._nodes[j].order, self._zkey[i + step]):
            step >>= 1
          if step > 0:
            self._nodes[j].npar = self._nodes[j].npar + step
            i = i + step
        i = i + 1
      # now close the remaining open nodes
      while j >= 0:
        self._nodes[j].npar = i - self._nodes[j].first
        j = self._nodes[j].parent
        
        
  @cython.boundscheck(False)
  cdef node_t _create_child(self, intptr_t first_par, intptr_t parent) nogil:
    # creates a child of parent from first_par, returns the new child */
    self._nodes[self.used].first = first_par
    self._nodes[self.used].npar = 1
    self._nodes[self.used].parent = parent
    self._nodes[self.used].child_length = 0
    self._nodes[self.used].order = self._nodes[parent].order - 1
    #/* the lower bits of a sqkey is cleared off but I don't think it is necessary */
#    self._nodes[self.used].key = (self._zkey[first_par] >> (self._nodes[self.used].order * 3)) << (self._nodes[self.used].order * 3)
    # the above code causes problem when zorder is not an integral type
    # thus we do not clear the lower bits.
    self._nodes[self.used].key = self._zkey[first_par]
    self._nodes[parent].child[self._nodes[parent].child_length] = self.used
    self._nodes[parent].child_length = self._nodes[parent].child_length + 1
    if self._nodes[parent].child_length > 8:
      return -1
    cdef node_t rt = self.used
    self.used = self.used + 1
    if self.used == self.size:
      self._grow()
    return rt

  def __dealloc__(self):
    free(self._nodes)


