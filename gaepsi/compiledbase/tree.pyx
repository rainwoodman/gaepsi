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
    node.link[prefix] = NULL
  return node

cdef inline int ipos_get_prefix(ipos_t ipos, int depth) nogil:
    return \
      (((ipos[0] >> depth) & 1) << 0) | \
      (((ipos[1] >> depth) & 1) << 1) | \
      (((ipos[2] >> depth) & 1) << 2)

cdef inline int ipos_compare(ipos_t a, ipos_t b, int bits) nogil:
  cdef int depth
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

cdef void node_make_argind_r(Node * node, npy_intp * indices, 
            npy_intp * next, npy_intp * iter) nogil:
  """ returns number of items added to indices """
  cdef npy_intp p
  if node.link[0]:
    for prefix in range(8):
      node_make_argind_r(node.link[prefix], indices, next, iter)
    node.first = node.link[0].first
  else:
    p = node.first
    node.first = iter[0]
    while p != -1:
      indices[iter[0]] = p
      p = next[p]
      iter[0] = iter[0] + 1

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
  
cdef Node * node_AABB_test_r(Node * node,
      npy_int64 a[3], npy_int64 b[3], npy_int64 c[3], npy_int64 d[3], 
      npy_int64 o[3], npy_int64 r,
      Node * head, Node ** next) nogil:

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
  if testvalue == 0:
    return head

  cdef npy_int64 nesta[3]
  cdef npy_int64 nestb[3]
  cdef int ax
  if testvalue == 1 or testvalue == 3 or spherevalue == 1:
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
               nesta, nestb, c, d, o, r, head, next)
      return head
  # either testvalue == 2, the entire node is inside,
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
  cdef numpy.ndarray scratch_index # used building the tree
  cdef numpy.ndarray scratch_pos  # used building the tree
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

  def __init__(self, pos, min=None, ptp=None, splitthresh=None, nbits=21):
    if min is None: min = numpy.min(pos, axis=0)
    if ptp is None: ptp = numpy.ptp(pos, axis=0)

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
    self.indices = numpy.zeros(len(pos), dtype='intp')
    self._indices = <npy_intp*> self.indices.data

    self.scratch_index = numpy.empty(self.splitthresh, dtype='intp')
    self.scratch_pos = numpy.empty(self.splitthresh, dtype=('f8', 3))

    self.num_splits = 0
    self.num_inserts = 0
    for d in range(3):
      self.ZERO[d] = 0
      self.LIMIT[d] = self.IPOS_LIMIT

    self._root.depth = self.IPOS_NBITS - 1

    cdef npy_intp * next = <npy_intp*>PyMem_Malloc(sizeof(npy_intp) * self.indices.shape[0])
    cdef npy_intp i
    for i in range(self.indices.shape[0]):
      next[i] = -1

    self.pos = pos
    cdef npy_intp chunksize = 65536
    cdef npy_intp * index = <npy_intp*>PyMem_Malloc(sizeof(npy_intp) * chunksize)
    cdef numpy.ndarray chunk

    for start in range(0, len(pos), chunksize):
       chunk = numpy.array(pos[start:start+chunksize], order='C', 
               copy=False, dtype=numpy.float64)
       for i in range(chunk.shape[0]):
         index[i] = start + i
       self.insert(self._root, chunk.shape[0], index, 
            <double (*)[3]> chunk.data, next)

    PyMem_Free(index)
    node_update_size_r(self._root)
    cdef npy_intp iter = 0 
    node_make_argind_r(self._root, self._indices, next, &iter)
    PyMem_Free(<void*>next)
    iter = 0
    node_collect(self._root, NULL, &iter)
    self.nodeptr = numpy.zeros(iter, 'intp')
    self._nodeptr = <Node **> self.nodeptr.data
    iter = 0
    node_collect(self._root, self._nodeptr, &iter)
    
  def __dealloc__(self):
    node_free_r(self._root)
    self._root = NULL

  def query(self, center, halfsize, exclude=0):
    """ return the index of particles that are almost within the given box
        this works only on one box. 
        if exclude > 0, then particles that are within this radius will 
        be very likely excluded. notice that if tree.invptp is
        anisotripic, the behavior of exclude is undefined.
    """
    center = numpy.asarray(center)
    halfsize = numpy.asarray(halfsize)

    cdef numpy.ndarray c = self.array_to_ipos(center - halfsize, 0, numpy.int64)
    cdef numpy.ndarray d = self.array_to_ipos(center + halfsize, 0, numpy.int64)
    cdef numpy.ndarray o = self.array_to_ipos(center, 1, numpy.int64)
    cdef npy_int64 r
    if exclude <= 0: 
      r = -1
    else:
      r = exclude * self.invptp[0] * self.IPOS_LIMIT

    cdef Node ** next = <Node**>PyMem_Malloc(sizeof(Node*) * self.nodeptr.shape[0])
    memset(next, 0, sizeof(Node*) * self.nodeptr.shape[0])

    cdef Node * head = node_AABB_test_r(self._root, 
            self.ZERO, self.LIMIT, <npy_int64*>c.data, <npy_int64*>d.data, 
            <npy_int64*>o.data, r, NULL, next)

    # first count total 
    cdef Node * p = head
    cdef npy_intp N = 0
    while p:
      N += p.size
      p = next[p.id]

    cdef numpy.ndarray result = numpy.empty(N, dtype='intp')
    cdef npy_intp * _result = <npy_intp*>result.data

    # then copy them over
    p = head
    i = 0
    while p:
      memcpy(&_result[i], &self._indices[p.first], p.size * sizeof(npy_intp))
      i = i + p.size
      p = next[p.id]
    PyMem_Free(next)
    return result

  cdef void insert(self, Node * node, npy_intp n, npy_intp * index, 
       double (* pos)[3], npy_intp * next):
    cdef npy_intp i
    cdef npy_int64 a[3]
    cdef npy_int64 lasta[3]
    cdef Node * lastnode = node
    cdef npy_int64 unmatch
    for i in range(0, n, 1):
      for d in range(3):
        lasta[d] = a[d]
      self.floating_to_ipos(pos[i], a, 1)
      while i != 0:
        for d in range(3):
          unmatch = (lasta[d] ^ a[d]) >> (lastnode.depth + 1)
          if unmatch: break
        if unmatch:
          lastnode = lastnode.up
        else:
          break
        if lastnode == node: break
      lastnode = self.node_insert_r(lastnode, index[i], a, next)
      #lastnode = self.node_insert_r(node, index[i], a, next)

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

  cdef void floating_to_ipos(self, floating x[3], npy_int64 ipos[3], int mode) nogil:
    cdef floating flt
    for d in range(3):
      flt = (x[d] - self._min[d]) * self._invptp[d]
      if mode == 0:
        if flt < 0: flt = 0
        if flt > 1: flt = 1
      else:
        flt = flt - floor(flt)
       
      ipos[d] = <npy_int64>(flt * self.IPOS_LIMIT)
      if ipos[d] == self.IPOS_LIMIT: ipos[d] = self.IPOS_LIMIT - 1

  cdef int cmp_pos(self, floating x[3], floating y[3]) nogil:
    # this is unused !
    cdef npy_int64 a[3]
    cdef npy_int64 b[3]
    self.floating_to_ipos(x, a, 1)
    self.floating_to_ipos(y, b, 1)
    return ipos_compare(a, b, self.IPOS_NBITS)

  cdef void node_split(self, Node * node, npy_intp * next):
    cdef npy_intp * _index = <npy_intp*> self.scratch_index.data
    self.num_splits = self.num_splits + 1
    for prefix in range(8):
      node.link[prefix] = node_alloc()
      node.link[prefix].up = node
      node.link[prefix].prefix = prefix
      node.link[prefix].depth = node.depth - 1

    p = node.first
    cdef npy_intp k = 0
    while p != -1:
      _index[k] = p
      k = k + 1
      p = next[p]

    node.first = -1
    node.size = 0
    numpy.PyArray_CopyInto(self.scratch_pos, numpy.PyArray_Take(self.pos, self.scratch_index, 0))
#    numpy.PyArray_TakeFrom(self.pos, self.scratch_index, 0, self.scratch_pos,
#             numpy.NPY_WRAP)

    self.insert(node, k, _index, <double (*)[3]>self.scratch_pos.data, next)
    
  cdef Node * node_insert_r(self, Node * node, npy_intp i,
          ipos_t a, npy_intp * next):
    """
      pos[i] and next[i] are the position and next pointer of i-th par.
    """
    self.num_inserts = self.num_inserts + 1
    if node.link[0]:
      prefix = ipos_get_prefix(a, node.depth)
      return self.node_insert_r(node.link[prefix], i, a, next)
    else:
      if node.size >= self.splitthresh and node.depth > 0:
        self.node_split(node, next)
        return self.node_insert_r(node, i, a, next)
      else:
        next[i] = node.first
        node.first = i
        node.size = node.size + 1
        return node
