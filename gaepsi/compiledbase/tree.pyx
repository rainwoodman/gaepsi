#cython: embedsignature=True
#cython: cdivision=True
import numpy
from cpython.mem cimport PyMem_Malloc, PyMem_Free
cimport numpy
from numpy cimport npy_intp, npy_int64
from cython cimport floating
from libc.string cimport memcpy, memset
DEF IPOS_NBITS = 20
DEF IPOS_LIMIT = (1L << IPOS_NBITS)

cdef packed struct Node:
  Node * up
  Node * link [8]
  int prefix
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
      cdef npy_int64 size = IPOS_LIMIT << 1
      while node:
        size >>= 1
        node = node.up
      return size
  property width:
    def __get__(self):
      return self.iwidth / self.tree.invptp / IPOS_LIMIT

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
      return self.icorner / self.tree.invptp / IPOS_LIMIT + self.tree.min

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

cdef int ipos_get_prefix(npy_int64 ipos[3], int depth) nogil:
    cdef int bit = (IPOS_NBITS - 1 - depth)
    return \
      (((ipos[0] >> bit) & 1) << 0) | \
      (((ipos[1] >> bit) & 1) << 1) | \
      (((ipos[2] >> bit) & 1) << 2)

cdef int ipos_compare(npy_int64 a[3], npy_int64 b[3]) nogil:
  cdef int depth
  for depth in range(IPOS_NBITS):
    prefix_a = ipos_get_prefix(a, depth);
    prefix_b = ipos_get_prefix(b, depth);
    if prefix_a < prefix_b:
      return -1
    if prefix_a > prefix_b:
      return 1
  return 0

cdef void node_insert_r(Node * node, npy_intp i, int depth,
        npy_int64 (*pos)[3], npy_intp * next, int splitthresh) nogil:
  """
    pos[i] and next[i] are the position and next pointer of i-th par.
  """
  cdef npy_intp p
  cdef npy_intp q

  if node.link[0]:
    prefix = ipos_get_prefix(pos[i], depth)
    node_insert_r(node.link[prefix], i, depth + 1, pos, next, splitthresh)
  else:
    if node.size >= splitthresh and depth < IPOS_NBITS - 1:
      # split
      with gil:
        for prefix in range(8):
          node.link[prefix] = node_alloc()
          node.link[prefix].up = node
          node.link[prefix].prefix = prefix
      p = node.first
      while p != -1:
        q = next[p]
        node_insert_r(node, p, depth, pos, next, splitthresh)
        p = q
      node_insert_r(node, i, depth, pos, next, splitthresh)
      node.first = -1
      node.size = 0
    else:
      next[i] = node.first
      node.first = i
      node.size = node.size + 1

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

cdef Node * node_AABB_test_r(Node * node,
      npy_int64 a[3], npy_int64 b[3], npy_int64 c[3], npy_int64 d[3], 
      Node * head, Node ** next) nogil:

  if node.size == 0: return head
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
  if testvalue == 1 or testvalue == 3:
    # c, d is not fully covered by node
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
               nesta, nestb, c, d, head, next)
      return head
  # either testvalue == 2, the entire node is inside,
  # or the node is external
  next[node.id] = head
  return node

cdef class Tree:
  cdef readonly numpy.ndarray indices
  cdef readonly numpy.ndarray invptp
  cdef readonly numpy.ndarray min
  cdef numpy.ndarray nodeptr
  cdef double * _invptp
  cdef double * _min
  cdef readonly int splitthresh
  cdef npy_intp * _indices
  cdef Node * _root
  cdef Node ** _nodeptr
  cdef npy_int64 ZERO[3]
  cdef npy_int64 LIMIT[3]

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
    for d in range(3):
      self.ZERO[d] = 0
      self.LIMIT[d] = IPOS_LIMIT

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

  def __init__(self, pos, min=None, ptp=None, splitthresh=None):
    if min is None: min = numpy.min(pos, axis=0)
    if ptp is None: ptp = numpy.ptp(pos, axis=0)
    if splitthresh is None:
      splitthresh = sizeof(Node) / 24 * 2
    if splitthresh <= 1: splitthresh = 2
    self.splitthresh = splitthresh
    self.min[:] = min
    ptp[ptp <= 0] = 1.0
    self.invptp[:] = 1. / ptp
    self.indices = numpy.zeros(len(pos), dtype='intp')
    self._indices = <npy_intp*> self.indices.data


    cdef npy_intp * next = <npy_intp*>PyMem_Malloc(sizeof(npy_intp) * self.indices.shape[0])
    cdef npy_intp i
    for i in range(self.indices.shape[0]):
      next[i] = -1

    cdef numpy.ndarray chunk = self.array_to_ipos(pos)
    self.build_chunk(0, len(chunk), <npy_int64 (*)[3]> chunk.data, next)

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

  def query(self, center, size):
    """ return the index of particles that are almost within the given box
        this works only on one box. 
        center
    """
    center = numpy.asarray(center)
    size = numpy.asarray(size)

    cdef numpy.ndarray c = self.array_to_ipos(center - size * 0.5)
    cdef numpy.ndarray d = self.array_to_ipos(center + size * 0.5)
    
    cdef Node ** next = <Node**>PyMem_Malloc(sizeof(Node*) * self.nodeptr.shape[0])
    memset(next, 0, sizeof(Node*) * self.nodeptr.shape[0])

    cdef Node * head = node_AABB_test_r(self._root, 
            self.ZERO, self.LIMIT, <npy_int64*>c.data, <npy_int64*>d.data, 
            NULL, next)

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

  cdef void build_chunk(self, npy_intp start, npy_intp n, 
              npy_int64 (*pos)[3], npy_intp * next) nogil:
    cdef npy_intp i
    for i in range(n):
      node_insert_r(self._root, start + i, 0, 
             pos - start, next, self.splitthresh)
     
  cdef numpy.ndarray array_to_ipos(self, numpy.ndarray array):
    cdef numpy.ndarray result = numpy.empty_like(array, dtype='int64')
    cdef numpy.ndarray flt = (array - self.min) * self.invptp
    flt.clip(0, 1, out=flt)
    result[...] = flt * IPOS_LIMIT
    result.clip(0, IPOS_LIMIT - 1, out=result)
    return result

  cdef void floating_to_ipos(self, floating x[3], npy_int64 ipos[3]) nogil:
    cdef floating flt
    for d in range(3):
      flt = (x[d] - self._min[d]) * self._invptp[d]
      if flt < 0: flt = 0
      if flt > 1: flt = 1
      ipos[d] = <npy_int64>(flt * IPOS_LIMIT)
      if ipos[d] == IPOS_LIMIT: ipos[d] = IPOS_LIMIT - 1

  cdef int cmp_pos(self, floating x[3], floating y[3]) nogil:
    cdef npy_int64 a[3]
    cdef npy_int64 b[3]
    self.floating_to_ipos(x, a)
    self.floating_to_ipos(y, b)
    return ipos_compare(a, b)

