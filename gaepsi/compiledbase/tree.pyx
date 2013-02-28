#cython: embedsignature=True
#cython: cdivision=True

import numpy
from cpython.mem cimport PyMem_Malloc, PyMem_Free
cimport numpy
from numpy cimport npy_int64, npy_int32
from libc.string cimport memcpy, memset
from libc.math cimport floor
from cython cimport floating
# do not use npy_intp. it was typedefed to int
# and fucks everything up with fused types.

from libc.stdint cimport intptr_t as npy_intp
from threading import Thread

numpy.import_array()

ctypedef npy_int64 (* ipos64ptr)[3]
ctypedef npy_int32 (* ipos32ptr)[3]
ctypedef npy_int64 * npy_int64ptr
ctypedef npy_int32 * npy_int32ptr
ctypedef fused ipos_ptr:
  ipos64ptr
  ipos32ptr
ctypedef fused ipos_t:
  npy_int64ptr
  npy_int32ptr

cdef packed struct Node:
  Node * up
  Node * link [8]
  short int prefix
  short int depth
  int id
  npy_intp first
  npy_intp size

cdef class TreeNode:
  cdef Node * node
  cdef readonly Tree tree
  property size:
    def __get__(self):
      return self.node.size
  property first:
    def __get__(self):
      return self.node.first
  property prefix:
    def __get__(self):
      return self.node.prefix
  property up:
    def __get__(self):
      cdef TreeNode rt
      if self.node.up:
        rt = TreeNode(self.tree)
        rt.node = self.node.up
        return rt
      else:
        return None
  property link:
    def __get__(self):
      if self.node.link[0]:
        return [self[i] for i in range(8)]
      else: 
        return []

  property iwidth:
    def __get__(self):
      cdef Node * node = self.node
      cdef npy_int64 size = self.tree.IPOS_LIMIT << 1
      while node:
        size >>= 1
        node = node.up
      return size
  property width:
    def __get__(self):
      return self.iwidth / self.tree.invptp / self.tree.IPOS_LIMIT

  property icorner:
    def __get__(self):
      cdef npy_int64 iwidth = self.iwidth
      cdef Node * node = self.node
      cdef numpy.ndarray icorner = numpy.zeros(3, dtype='int64')
      cdef npy_int64 * _icorner = <npy_int64*> icorner.data
      while node.up:
        for d in range(3):
          if node.prefix & (1 << d):
            _icorner[d] += iwidth
        iwidth <<= 1
        node = node.up
      return icorner
  property corner:
    def __get__(self):
      return self.icorner / self.tree.invptp / self.tree.IPOS_LIMIT + self.tree.min

  property pos:
    def __get__(self):
      return self.tree.pos[self.tree.indices[self.first:self.first+self.size]]

  property ipos:
    def __get__(self):
      return self.tree.array_to_ipos(self.pos, 1, numpy.int64)

  property key:
    def __get__(self):
      return self.tree.ipos_to_string(self.ipos)

  def check(self):
    cdef npy_int64 a[3]
    cdef npy_int64 b[3]
    bad = []
    cdef numpy.ndarray ipos = self.ipos
    for i in range(len(ipos)):
      p = ipos[i]
      a[0] = ipos[i, 0]
      a[1] = ipos[i, 1]
      a[2] = ipos[i, 2]
      if i < len(ipos) - 1:
        b[0] = ipos[i+1, 0]
        b[1] = ipos[i+1, 1]
        b[2] = ipos[i+1, 2]
        if not ipos_compare(a, b, self.tree.IPOS_NBITS) <= 0:
          bad.append((i, i+1))
      if not node_contains_ipos(self.node, a):
        bad.append(i)
    return bad


  def __cinit__(self, Tree tree):
    self.tree = tree

  def __getitem__(self, index):
    cdef TreeNode rt

    if numpy.isscalar(index):
      if index >= 0 and index < 8 and self.node.link[index] != NULL:
        rt = TreeNode(self.tree)
        rt.node = self.node.link[index]
        return rt
      else:
        return None
    elif len(index) == 1:
      return self[index[0]]
    else:
      rt = self[index[0]]
      if rt is not None:
        return rt[index[1:]]
      else:
        return None
  def __str__(self):
    if self.node:
      return '%d (size=%d, first=%d, width=%s corner=%s)' % (self.node.id, self.size, 
                          self.first, str(self.width), str(self.corner))
    else:
      return '<null>'
  def __repr__(self):
    return str(self)

cdef void node_free_r(Node * node):
  if node.link[0]:
    for i in range(8):
      node_free_r(node.link[i])
  PyMem_Free(node)

cdef Node * node_alloc():
  node = <Node*> PyMem_Malloc(sizeof(Node))
  node.size = 0
  node.first = -1
  for prefix in range(8):
    node.id = <int>node
    node.link[prefix] = NULL
  return node

cdef inline int ipos_get_prefix(ipos_t ipos, int depth) nogil:
    return \
      (((ipos[0] >> depth) & 1) << 0) | \
      (((ipos[1] >> depth) & 1) << 1) | \
      (((ipos[2] >> depth) & 1) << 2)

cdef inline int ipos_compare(ipos_t a, ipos_t b, int bits) nogil:
  cdef int depth
  cdef int prefix_a
  cdef int prefix_b
  for depth in range(bits - 1, -1, -1):
    prefix_a = ipos_get_prefix(a, depth);
    prefix_b = ipos_get_prefix(b, depth);
    if prefix_a < prefix_b:
      return -1
    if prefix_a > prefix_b:
      return 1
  return 0


cdef npy_intp node_update_size_r(Node * node) nogil:
  cdef npy_intp size = 0
  if node.link[0]:
    for prefix in range(8):
      size += node_update_size_r(node.link[prefix])
    node.size = size
  return node.size

cdef void node_collect(Node * node, Node ** collect, npy_intp * iter) nogil:
  """ collect all nodes into an array at collect.

      if collect is NULL, just update *iter to the needed size
  """
  node.id = iter[0]
  if collect: collect[iter[0]] = node
  iter[0] = iter[0] + 1
  if node.link[0]:
    for prefix in range(8):
      node_collect(node.link[prefix], collect, iter)
    
cdef int AABBtest(npy_int64 a[3], npy_int64 b[3], npy_int64 c[3], npy_int64 d[3]) nogil:
  """ a b are corners of the first box, c, d are coners of the second box
      b > a and d > c
      open on b and d, close on a and c.
      returns 0 if not intersect
      > 0 intersects
      1 partial
      2 ab is fully in cd
      3 cd is fully in ab
  """
  cdef int good_edge_2 = 0
  cdef int good_edge_3 = 0
  for ax in range(3):
    if a[ax] > d[ax] or c[ax] > b[ax]: return 0
    if a[ax] >= c[ax] and b[ax] <= d[ax]: 
      good_edge_2 = good_edge_2 + 1
    if c[ax] >= a[ax] and d[ax] <= b[ax]: 
      good_edge_3 = good_edge_3 + 1
  if good_edge_2 == 3: return 2
  if good_edge_3 == 3: return 3
  return 1
cdef int spheretest(npy_int64 a[3], npy_int64 b[3], npy_int64 o[3], npy_int64 r) nogil:
  """ returns 0 if all of the corner of ab is outside of sphere of radius r
      centering at o. [possible non-overlap]
      returns 1 if any of the corner of box is in the sphere [definitely overlap]
      returns 2 if the box ab is fully in the sphere. [definitely inside]
  """
  cdef npy_int64 p[3]
  cdef int i
  cdef int d
  cdef double x
  cdef double sum
  cdef int outsidevalue = 0
  for i in range(8):
    sum = 0
    for d in range(3):
      if i & (1 << d):
        p[d] = b[d]
      else:
        p[d] = a[d]
      p[d] = p[d] - o[d]
      if p[d] >= r or p[d] <= -r: 
        sum = 10.
        break
      else:
        x = (p[d] / <double> r) 
        sum = sum + x * x
    if sum >= 1: 
      outsidevalue = outsidevalue + 1
  if outsidevalue == 8: return 0
  if outsidevalue == 0: return 2
  return 1
  
cdef int node_contains_ipos(Node * node, npy_int64 a[3]) nogil:
  cdef int d
  while node.up:
    if ipos_get_prefix(a, node.depth + 1) != node.prefix:
      return 0
    node = node.up
  return 1

cdef int resolution_test(npy_int64 a[3], npy_int64 b[3], 
           npy_int64 o[3], double resolution) nogil:
  """ returns 1 if the open angle of ab is bigger than resolution
  """
  cdef double l = 0.0
  cdef double r = 0.0
  cdef npy_int64 c
  cdef npy_int64 w
  for d in range(3):
    w = (b[d] - a[d])
    c = a[d] - o[d] + (w >> 1) 
    l = l + 1.0 * w * w
    r = r + 1.0 * c * c
  return l >= r * resolution * resolution 

cdef Node * node_resolution_test_r(Node * node, npy_int64 width, npy_int64 widthlimit, 
        Node * head, Node ** next, int * flags) nogil:
  if node.size == 0: return head
  flags[node.id] = width > widthlimit
  if node.link[0] and flags[node.id]:
    width >>= 1
    for prefix in range(8):
      head = node_resolution_test_r(node.link[prefix],
               width, widthlimit, head, next, flags)
    return head
  else:
    next[node.id] = head
    return node
    
cdef Node * node_AABB_test_r(Node * node,
      npy_int64 a[3], npy_int64 b[3], npy_int64 c[3], npy_int64 d[3], 
      npy_int64 o[3], npy_int64 r, double resolution,
      Node * head, Node ** next, int * flags) nogil:

  if node.size == 0: return head
  cdef int spherevalue = 0 
  # take the possible non-overlap branch if no exclusion zone
  if r > 0:
    spherevalue = spheretest(a, b, o, r)
  if spherevalue == 2: 
    # definitely inside exclusion zone
    return head

  cdef int testvalue = AABBtest(a, b, c, d)
#  if testvalue == 2:
#    with gil:
#       print 'AABB r:', node.id, testvalue
#       print 'AABB a:', a[0], a[1], a[2]
#       print 'AABB b:', b[0], b[1], b[2]
#       print 'AABB c:', c[0], c[1], c[2]
#       print 'AABB d:', d[0], d[1], d[2]

  # definitely outside the includsive zone
  if testvalue == 0:
    return head

  cdef npy_int64 nesta[3]
  cdef npy_int64 nestb[3]
  if resolution > 0:
    flags[node.id] = resolution_test(a, b, o, resolution)
  else:
    flags[node.id] = 1
  cdef int ax
  if (testvalue == 1 or testvalue == 3 or spherevalue == 1 ) \
    or (testvalue == 2 and flags[node.id]):
    # c, d is not fully covered by node,
    # or cd is definitely overlapping the exclusion zone,
    # but not definitely fully inside
    if node.link[0]:
      for prefix in range(8):
        for ax in range(3):
          if prefix & (1 << ax):
            nesta[ax] = a[ax] + ((b[ax] - a[ax]) >> 1)
            nestb[ax] = b[ax]
          else:
            nesta[ax] = a[ax]
            nestb[ax] = a[ax] + ((b[ax] - a[ax]) >> 1)
        head = node_AABB_test_r(node.link[prefix],
               nesta, nestb, c, d, o, r, resolution, head, next, flags)
      return head
  # either the entire node is inside (value=2) and unresolved
  # or the node is external
  next[node.id] = head
  return node

cdef Node * node_AABB_test_r2(Node * node,
      npy_int64 a[3], npy_int64 b[3], npy_int64 c[3], npy_int64 d[3], 
      npy_int64 o[3], npy_int64 r, npy_int64 widthlimit,
      Node * head, Node ** next, int * flags) nogil:

  if node.size == 0: return head
  cdef int spherevalue = 0 
  # take the possible non-overlap branch if no exclusion zone
  if r > 0:
    spherevalue = spheretest(a, b, o, r)
  if spherevalue == 2: 
    # definitely inside exclusion zone
    return head

  cdef int testvalue = AABBtest(a, b, c, d)
#  if testvalue == 2:
#    with gil:
#       print 'AABB r:', node.id, testvalue
#       print 'AABB a:', a[0], a[1], a[2]
#       print 'AABB b:', b[0], b[1], b[2]
#       print 'AABB c:', c[0], c[1], c[2]
#       print 'AABB d:', d[0], d[1], d[2]

  # definitely outside the includsive zone
  if testvalue == 0:
    return head

  cdef npy_int64 nesta[3]
  cdef npy_int64 nestb[3]

  flags[node.id] = b[0] - a[0] > widthlimit
  cdef int ax
  if (testvalue == 1 or testvalue == 3 or spherevalue == 1 ) \
    or (testvalue == 2 and flags[node.id]):
    # c, d is not fully covered by node,
    # or cd is definitely overlapping the exclusion zone,
    # but not definitely fully inside
    if node.link[0]:
      for prefix in range(8):
        for ax in range(3):
          if prefix & (1 << ax):
            nesta[ax] = a[ax] + ((b[ax] - a[ax]) >> 1)
            nestb[ax] = b[ax]
          else:
            nesta[ax] = a[ax]
            nestb[ax] = a[ax] + ((b[ax] - a[ax]) >> 1)
        head = node_AABB_test_r2(node.link[prefix],
               nesta, nestb, c, d, o, r, widthlimit, head, next, flags)
      return head
  # either the entire node is inside (value=2) and unresolved
  # or the node is external
  next[node.id] = head
  return node

cdef class Tree:
  cdef readonly numpy.ndarray indices
  cdef readonly numpy.ndarray invptp
  cdef readonly numpy.ndarray min
  cdef readonly int IPOS_NBITS
  cdef readonly npy_int64 IPOS_LIMIT
  cdef numpy.ndarray nodeptr
  cdef double * _invptp
  cdef double * _min
  cdef readonly int splitthresh
  cdef npy_intp * _indices
  cdef Node * _root
  cdef Node ** _nodeptr
  cdef npy_int64 ZERO[3]
  cdef npy_int64 LIMIT[3]
  cdef readonly numpy.ndarray pos # reference to the positions
  cdef readonly npy_intp num_splits
  cdef readonly npy_intp num_inserts

  property root:
    def __get__(self):
      cdef TreeNode root = TreeNode(self)
      root.node = self._root
      return root

  def __cinit__(self):
    self.invptp = numpy.ones(3, dtype='f8')
    self.min = numpy.ones(3, dtype='f8')
    self._invptp = <double *> self.invptp.data
    self._min = <double *> self.min.data
    self._root = node_alloc()
    self._root.up = NULL
    for i in range(8):
      self._root.link[i] = NULL

  def __len__(self):
    return len(self.nodeptr)

  def __getitem__(self, index):
    cdef TreeNode rt
    if numpy.isscalar(index):
      rt = TreeNode(self)
      rt.node = self._nodeptr[index]
      return rt
    else:
      if isinstance(index, numpy.ndarray):
        return [self[ind] for ind in index]
      else:
        raise IndexError("index is not array or scalar")

  def __init__(self, min, ptp, splitthresh=None, nbits=21):

    self.IPOS_NBITS = nbits
    self.IPOS_LIMIT = (1L << self.IPOS_NBITS)

    if splitthresh is None:
      splitthresh = sizeof(Node) / 24 * 2
    if splitthresh <= 1: splitthresh = 2
    self.splitthresh = splitthresh
    self.min[:] = min
    self.invptp[:] = ptp
    self.invptp[self.invptp <= 0] = 1.0
    self.invptp[:] = 1.0 / self.invptp

    self.num_splits = 0
    self.num_inserts = 0
    for d in range(3):
      self.ZERO[d] = 0
      self.LIMIT[d] = self.IPOS_LIMIT

    self._root.depth = self.IPOS_NBITS - 1

  def build(self, pos, argsort=None):

    self.pos = pos
    self.indices = numpy.zeros(len(pos), dtype='intp')
    self._indices = <npy_intp*> self.indices.data
    if argsort is None:
      self.argsort(pos, out=self.indices)
    else:
      argsort(self, pos, self.indices)
    self._addchunk()
    self._postprocess()

  def merge(self, numpy.ndarray pos, numpy.ndarray A, numpy.ndarray B, numpy.ndarray out):
    assert A.dtype == numpy.dtype(numpy.intp)
    assert B.dtype == numpy.dtype(numpy.intp)
    assert out.dtype == numpy.dtype(numpy.intp)
    assert len(out) == len(pos)

    cdef int itemsize = pos.dtype.itemsize
    with nogil:
      if itemsize == 8:
        self._merge(<double *> pos.data, <npy_intp*>pos.strides, 
                 <npy_intp*> A.data, A.shape[0], <npy_intp*> B.data, B.shape[0], 
                 <npy_intp*> out.data)
      elif itemsize == 4:
        self._merge(<float*> pos.data, <npy_intp*>pos.strides, 
                 <npy_intp*> A.data, A.shape[0], <npy_intp*> B.data, B.shape[0], 
                 <npy_intp*> out.data)
    #print 'merge'
    #print self.ipos_to_string(self.array_to_ipos(pos[A], 1, numpy.int64))
    #print self.ipos_to_string(self.array_to_ipos(pos[B + len(A)], 1, numpy.int64))
    #print self.ipos_to_string(self.array_to_ipos(pos[out], 1, numpy.int64))
  def argsort(self, numpy.ndarray pos, out=None):
    cdef numpy.ndarray arg1
    if out is None:
      out = numpy.empty(len(pos), numpy.intp)
    else:
      assert len(out) == len(pos)
      assert out.dtype == numpy.dtype(numpy.intp)
    arg1 = out
    cdef npy_intp i
    cdef int itemsize = pos.dtype.itemsize
    with nogil:
      for i in range(pos.shape[0]):
        (<npy_intp *> arg1.data)[i] = i
      if itemsize == 8:
        self._argsort(<double *> pos.data, <npy_intp*>pos.strides, <npy_intp*> arg1.data, pos.shape[0])
      elif itemsize == 4:
        self._argsort(<float *> pos.data, <npy_intp*>pos.strides, <npy_intp*> arg1.data, pos.shape[0])
    #print out.shape, self.ipos_to_string(self.array_to_ipos(pos[out], 1, numpy.int64))
    return out

  def _postprocess(self):
    node_update_size_r(self._root)
    cdef npy_intp iter = 0 
    iter = 0
    node_collect(self._root, NULL, &iter)
    self.nodeptr = numpy.zeros(iter, 'intp')
    self._nodeptr = <Node **> self.nodeptr.data
    iter = 0
    node_collect(self._root, self._nodeptr, &iter)

  def _addchunk(self):
    cdef npy_intp i
    cdef positer iter = positer(self.pos, self.indices, self.splitthresh * 4, 'f8')
    i = 0
    node = self._root
    cdef npy_int64 a[3]
    cdef TreeNode tn
    tn = TreeNode(self)
    tn.node = node
    #print 'root',  tn
    while i < self.indices.shape[0]:
      self.floating_to_ipos(iter.get(i), a, 1)
      tn.node = node
      #print 'visiting node', tn
      while node and not node_contains_ipos(node, a):
        node.size = i - node.first
        node = node.up
        tn.node = node
        #print 'close node', tn
      assert node
      while node.link[0]:
        prefix = ipos_get_prefix(a, node.depth)
        node = node.link[prefix]
      if node.size >= self.splitthresh and node.depth > 0:
        self.num_splits = self.num_splits + 1
        tn.node = node
        #print 'split node', tn
        for prefix in range(8):
          node.link[prefix] = node_alloc()
          node.link[prefix].up = node
          node.link[prefix].prefix = prefix
          node.link[prefix].depth = node.depth - 1
          node.link[prefix].first = node.first
        i = node.first
        if i < 0:
          print 'i assertion failed'
        continue
      else:
        tn.node = node
        #print 'grow node', tn
        self.num_inserts = self.num_inserts + 1
        if node.size == 0: node.first = i
        node.size = node.size + 1
        i = i + 1

    while node:
      node.size = i - node.first
      node = node.up

  def __dealloc__(self):
    node_free_r(self._root)
    self._root = NULL

  def centerofmass(self, mass=1.):
    result = numpy.empty(len(self), dtype=('f8', 3))
    for d in range(3):
      result[:, d] = self.count(self.pos[:, d] * mass)
    result /= self.count(weights=mass)[:, None]
    return result

  def count(self, weights=None):
    """ weights can be 
         None or size: number of particles
         width: width of a node
         volume: volume of a node
         iwidth: iwidth of a node
         prefix: 
         first: 
         depth: 
    
         array of length(self.pos): 
    """
    cdef numpy.ndarray result 
    cdef numpy.ndarray cweights
    cdef Node * node
    if weights is None or weights == 'size':
      result = numpy.empty(shape=len(self), dtype='i8')
      for i in range(len(self)):
        (<npy_int64 *> result.data)[i] = self._nodeptr[i].size
    elif weights == 'first':
      result = numpy.empty(shape=len(self), dtype='i8')
      for i in range(len(self)):
        (<npy_int64 *> result.data)[i] = self._nodeptr[i].first
    elif weights == 'depth':
      result = numpy.empty(shape=len(self), dtype='i4')
      for i in range(len(self)):
        (<npy_int32 *> result.data)[i] = self._nodeptr[i].depth
    elif weights == 'prefix':
      result = numpy.empty(shape=len(self), dtype='i4')
      for i in range(len(self)):
        (<npy_int32 *> result.data)[i] = self._nodeptr[i].prefix
    elif weights == 'width':
      result = numpy.empty(shape=len(self), dtype='f8')
      for i in range(len(self)):
        (<double *> result.data)[i] = \
         1.0 / self.invptp[0] / \
         (1 << (self._root.depth - self._nodeptr[i].depth))
    elif weights == 'volume':
      result = numpy.empty(shape=len(self), dtype='f8')
      for i in range(len(self)):
        (<double *> result.data)[i] = \
          1.0  / self.invptp[0] / self.invptp[1] / self.invptp[2] \
          / (1 << (self._root.depth - self._nodeptr[i].depth)) \
          / (1 << (self._root.depth - self._nodeptr[i].depth))  \
          / (1 << (self._root.depth - self._nodeptr[i].depth)) 
    elif weights == 'iwidth':
      result = numpy.empty(shape=len(self), dtype='f8')
      for i in range(len(self)):
        (<double *> result.data)[i] = \
         (self.IPOS_LIMIT >> (self._root.depth - self._nodeptr[i].depth))
    elif numpy.isscalar(weights):
      return self.count(None) * weights
    else:
      result = numpy.zeros(shape=len(self), dtype='f8')
      cweights = numpy.ascontiguousarray(weights, dtype='f8')
      assert cweights.shape[0] == len(self.pos)
      self.node_count_r(self._root, <double*>cweights.data, 
               <double*>result.data)
    return result

  cdef floating node_count_r(self, Node * node, floating * weights, double * out):
    cdef npy_intp i 
    if node.link[0]:
      for prefix in range(8):
        out[node.id] += self.node_count_r(node.link[prefix], weights, out)
      return out[node.id]
    else:
      for i in range(0, node.size, 1):
        out[node.id] += weights[self._indices[node.first + i]]
      return out[node.id]

  def query(self, center, halfsize, exclude=0, resolution=None):
    """ return the index of particles that are almost within the given box
        this works only on one box. 
        if exclude > 0, then particles that are within this radius will 
        be very likely excluded. notice that if tree.invptp is
        anisotripic, the behavior of exclude is undefined.

        if resolution is a positive number (with 0 ~ 1, smaller == finer)
        then returns two arrays, 
          index of particles possibly in the region
        and
          index of nodes possibly in the region, if they are unresolved.
        when center is None,
        do not query a region, but find all objects that are resolved by resolution * boxsize.
    """
    cdef double res
    cdef Node ** next = <Node**>PyMem_Malloc(sizeof(Node*) * self.nodeptr.shape[0])
    memset(next, 0, sizeof(Node*) * self.nodeptr.shape[0])
  
    cdef int * flags = <int*>PyMem_Malloc(sizeof(int) * self.nodeptr.shape[0])
  
    if resolution is None:
      res = 0
    else:
      res = resolution
    cdef Node * head
    cdef numpy.ndarray c
    cdef numpy.ndarray d
    cdef numpy.ndarray o
    cdef npy_int64 r
    if center is None:
      with nogil:
         head = node_resolution_test_r(self._root, self.IPOS_LIMIT, 
                        <npy_int64>(res * self.IPOS_LIMIT), NULL,
                        next, flags)
    else:
      center = numpy.asarray(center)
      halfsize = numpy.asarray(halfsize)
  
      c = self.array_to_ipos(center - halfsize, 0, numpy.int64)
      d = self.array_to_ipos(center + halfsize, 0, numpy.int64)
      o = self.array_to_ipos(center, 1, numpy.int64)
      if exclude <= 0: 
        r = -1
      else:
        r = <npy_int64>(<double>exclude * self.invptp[0] * self.IPOS_LIMIT)
  
      with nogil:
        head = node_AABB_test_r2(self._root, 
              self.ZERO, self.LIMIT, <npy_int64*>c.data, <npy_int64*>d.data, 
                       <npy_int64*>o.data, r, 
                     <npy_int64>(res * self.IPOS_LIMIT), NULL,
                     next, flags)

    # first count total 
    cdef Node * p = head
    cdef npy_intp N = 0
    cdef npy_intp nodeN = 0
    cdef TreeNode tn = TreeNode(self)
    with nogil:
      while p:
        if flags[p.id] != 0:
          N += p.size
        else:
          nodeN += 1
        p = next[p.id]

    cdef numpy.ndarray result = numpy.empty(N + nodeN, dtype='intp')
    cdef npy_intp * _result = <npy_intp*>result.data
    cdef npy_intp * _resultNode = <npy_intp*>result.data + N

    # then copy them over
    p = head
    cdef npy_intp i = 0
    cdef npy_intp j = 0
    with nogil:
      while p:
        if flags[p.id] != 0:
          memcpy(&_result[i], &self._indices[p.first], p.size * sizeof(npy_intp))
          i = i + p.size
        else:
          _resultNode[j] = p.id
          j = j + 1
        p = next[p.id]

    PyMem_Free(flags)
    PyMem_Free(next)

    if resolution is not None:
      return result[:N], result[N:]
    else:
      return result

  cdef numpy.ndarray array_to_ipos(self, numpy.ndarray array, mode, dtype):
    """ mode == 0, clip
        mode == 1, wrap
        usually use wrap, only use clip for the input of .query
    """
    cdef numpy.ndarray result = numpy.empty_like(array, dtype=dtype)
    cdef numpy.ndarray flt = (array - self.min) * self.invptp
    if mode == 0:
      flt.clip(0, 1, out=flt)
    else:
      numpy.remainder(flt, 1.0, out=flt)
    result[...] = flt * self.IPOS_LIMIT
    result.clip(0, self.IPOS_LIMIT - 1, out=result)
    return result

  cdef void floating_to_ipos(self, floating x[3], ipos_t ipos, int mode) nogil:
    cdef double flt
    for d in range(3):
      flt = (x[d] - self._min[d]) * self._invptp[d]
      if mode == 0:
        if flt < 0: flt = 0
        if flt > 1: flt = 1
      else:
        flt = flt - floor(flt)
       
      ipos[d] = <npy_int64>(flt * self.IPOS_LIMIT)
      if ipos[d] == self.IPOS_LIMIT: ipos[d] = self.IPOS_LIMIT - 1

  cdef int floating_compare(self, floating * v, npy_intp * strides, npy_intp ia, npy_intp ib, int mode) nogil:
    cdef npy_int64 a[3]
    cdef npy_int64 b[3]
    cdef floating fa[3]
    cdef floating fb[3]
    cdef int d
    for d in range(3):
      fa[d] = (<floating*>(<char*> v + strides[0] * ia + strides[1] * d))[0]
      fb[d] = (<floating*>(<char*> v + strides[0] * ib + strides[1] * d))[0]
    self.floating_to_ipos(fa, a, 1)
    self.floating_to_ipos(fb, b, 1)
    return ipos_compare(a, b, self.IPOS_NBITS)

  cdef int _merge(self, floating *vl, npy_intp * strides, npy_intp * A, npy_intp Anum, npy_intp * B, npy_intp Bnum, npy_intp * C) nogil:
    cdef npy_intp i, j, k
    i = 0
    j = 0
    k = 0
    while i < Anum or j < Bnum:
      while i < Anum and (j == Bnum or self.floating_compare(vl, strides, A[i], B[j] + Anum, 1) <= 0):
        C[k] = A[i]
        k = k + 1
        i = i + 1
      while j < Bnum and (i == Anum or self.floating_compare(vl, strides, A[i], B[j] + Anum, 1) >= 0):
        C[k] = B[j] + Anum
        k = k + 1
        j = j + 1
    return 0

  cdef int _argsort(self, floating *vl, npy_intp * strides, npy_intp * tosort, npy_intp num) nogil:
    cdef npy_intp pmsave
    cdef npy_intp *pl
    cdef npy_intp *pr
    cdef npy_intp *stack[128]
    cdef npy_intp **sptr=stack
    cdef npy_intp *pm
    cdef npy_intp *pi
    cdef npy_intp *pj
    cdef npy_intp *pk
    cdef npy_intp vi
    cdef npy_intp tmp

    pl = tosort;
    pr = tosort + num - 1;

    while True:
        while pr - pl > 17:
            #/* quicksort partition */
            pm = pl + ((pr - pl) >> 1);
            if self.floating_compare(vl, strides, pm[0], pl[0], 1) < 0: tmp = pm[0]; pm[0] = pl[0]; pl[0] = tmp
            if self.floating_compare(vl, strides, pr[0], pm[0], 1) < 0: tmp = pr[0]; pr[0] = pm[0]; pm[0] = tmp
            if self.floating_compare(vl, strides, pm[0], pl[0], 1) < 0: tmp = pm[0]; pm[0] = pl[0]; pl[0] = tmp
            pmsave = pm[0]
            pi = pl
            pj = pr - 1
            tmp = pm[0]; pm[0] = pj[0]; pj[0] = tmp
            while True:
                pi = pi + 1
                while self.floating_compare(vl, strides, pi[0], pmsave, 1) < 0: pi = pi + 1
                pj = pj - 1
                while self.floating_compare(vl, strides, pmsave, pj[0], 1) < 0: pj = pj - 1
                if pi >= pj: break
                tmp = pi[0]; pi[0] = pj[0]; pj[0] = tmp
            pk = pr - 1
            tmp = pi[0]; pi[0] = pk[0]; pk[0] = tmp
            # push largest partition on stack
            if pi - pl < pr - pi:
                sptr[0] = pi + 1;
                sptr[1] = pr;
                sptr = sptr + 2
                pr = pi - 1;
            else:
                sptr[0] = pl;
                sptr[1] = pi - 1;
                sptr = sptr + 2
                pl = pi + 1;

        # insertion sort
        pi = pl + 1
        while pi <= pr:
            vi = pi[0]
            pmsave = vi
            pj = pi
            pk = pi - 1
            while pj > pl and self.floating_compare(vl, strides, pmsave, pk[0], 1) < 0:
                pj[0] = pk[0]
                pj = pj - 1
                pk = pk - 1
            pj[0] = vi;
            pi = pi + 1
        if sptr == stack: break

        sptr = sptr - 1
        pr = sptr[0]
        sptr = sptr - 1
        pl = sptr[0]

    return 0

  def pos_to_string(self, pos):
    return self.ipos_to_string(self.array_to_ipos(pos, 1, numpy.int64))

  def ipos_to_string(self, ipos):
    cdef npy_int64 a[3]
    cdef int j
    cdef numpy.ndarray s = numpy.empty(self.IPOS_NBITS, dtype='c')
    if len(ipos) == 0: return []
    if len(ipos.shape) != 1:
      return numpy.asarray([self.ipos_to_string(p) for p in ipos])
    a[0] = ipos[0]
    a[1] = ipos[1]
    a[2] = ipos[2]
    for j in range(self.IPOS_NBITS - 1, -1, -1):
      s[(self.IPOS_NBITS - j - 1)] = '01234567'[ipos_get_prefix(a, j)]
    return s.tostring()


#  cdef int cmp_pos(self, floating x[3], floating y[3]) nogil:
    # this is unused !
#    cdef npy_int64 a[3]
#    cdef npy_int64 b[3]
#    self.floating_to_ipos(x, a, 1)
#    self.floating_to_ipos(y, b, 1)
#    return ipos_compare(a, b, self.IPOS_NBITS)

cdef class positer:
  cdef readonly numpy.ndarray pos
  cdef readonly numpy.ndarray indices
  cdef readonly numpy.dtype dtype
  cdef readonly numpy.ndarray buffer
  cdef readonly npy_intp buffersize
  cdef readonly npy_intp start
  def __init__(self, pos, indices, buffersize, dtype):
    self.pos = pos
    self.indices = indices
    self.dtype = pos.dtype
    if buffersize > len(pos): buffersize = len(pos)
    self.buffer = numpy.empty(shape=(buffersize, 3), dtype=dtype)
    self.start = 0
    self.buffer[:] = pos[indices[:buffersize]]
    self.buffersize = buffersize

  cdef double * get(self, npy_intp cursor) nogil:
    cdef npy_intp size = 0
    cdef double * rt
    if cursor < self.start or cursor >= self.start + self.buffersize:
      self.start = cursor
      size = self.buffersize
      if cursor+size > self.pos.shape[0]:
        size = self.pos.shape[0] - cursor
      with gil:
        try:
          ind = self.indices[cursor:cursor+size]
          self.buffer[:size] = self.pos[ind]
        except ValueError as e:
          print e 
        #print 'update', cursor, ind, self.pos[ind], size
      rt = (<double (*)[3]> self.buffer.data) [0]
    else:
      rt = (<double (*)[3]> self.buffer.data)[ cursor - self.start]
    
    #with gil:
      #print 'returning', cursor, self.start, size, rt[0], rt[1], rt[2]
   
    return rt

