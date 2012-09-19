#cython: embedsignature=True
#cython: cdivision=True
import numpy
cimport numpy
cimport cpython
from ztree cimport Tree, node_t
from libc.stdint cimport *
from libc.stdlib cimport malloc, realloc, free
cimport npyiter
from libc.math cimport sqrt
from geometry cimport LiangBarsky
cimport fillingcurve
from fillingcurve cimport fckey_t, ipos_t

numpy.import_array()
cdef extern from 'zquery_internal.c':
  ctypedef struct Heap:
    int (*cmp_lt)(int i, int j, void* weight)
    void (*load)(int i, int j, void* data, void * weight)
    void * data
    void * weight
    size_t length
  void _siftdown(Heap * self, int startpos, int pos) nogil
  void _siftup(Heap * self, int pos) nogil
  void _heapify(Heap * self) nogil

cdef int cmp_lt(int i, int j, void * weight) nogil:
  cdef double * f = <double*>weight
  # NOTE: reversed order!
  return f[i] > f[j]
cdef void load(int i, int j, void * data, void * weight) nogil:
  cdef intptr_t * I = <intptr_t*>data
  cdef double * F = <double*>weight
  F[i] = F[j]
  I[i] = I[j]

cdef void makeheap(Heap * self, Query query) nogil:
  self.cmp_lt = cmp_lt
  self.load = load
  self.data = query._items + 1
  self.weight = query._weight + 1
  self.length = query.size

cdef class freeobj:
  cdef void * pointer
  def __dealloc__(self):
    free(self.pointer)
   
cdef numpy.ndarray construct_arrays(numpy.ndarray pointers, numpy.ndarray used, int type):
    """ convert pointers to array of numpy arrays """
    out = numpy.empty_like(pointers, dtype='object')
    iter = numpy.nditer([pointers, used, out], 
       op_flags= [['readonly'], ['readonly'], ['writeonly']],
          flags=['buffered', 'external_loop', 'refs_ok'], 
      casting='unsafe', 
      op_dtypes=['intp'] * 2 + ['object'])
    cdef npyiter.CIter citer
    cdef size_t size = npyiter.init(&citer, iter)
    cdef void * temp
    while size > 0:
      while size > 0:
        temp = (<void**>citer.data[0])[0]
        arr = numpy.PyArray_SimpleNewFromData(1, <numpy.intp_t*>citer.data[1], 
                type, temp)
        obj = freeobj()
        numpy.set_array_base(arr, obj)
        obj.pointer = temp
        (<void **>citer.data[2])[0] = <void *>arr
        cpython.Py_INCREF(arr)
        npyiter.advance(&citer)
        size = size - 1
      size = npyiter.next(&citer)
    if len(out.shape) == 0:
      return out[()]
    else: return out

cdef class Query:

  def __cinit__(self, int limit, bint weighted):
    """ if weighted is True, a weighted search is performed.
           limit is the total number of particles to return, the particles with
           less weight is returned.
        if weighted is False, limit is the initial size of the buffer,
           all particles are returned.
    """
    self._weighted = weighted
    if weighted:
      self.size = limit
      self.used = 0
      # NOTE: first (0th) item used by the Heap algorithm as a scratch.
      self._weight = (<double *> malloc(sizeof(double) * (self.size + 1)))
      self._items = (<intptr_t *> malloc(sizeof(intptr_t) * (self.size + 1)))
    else:
      self.size = limit
      self.used = 0
      self._weight = NULL
      self._items = <intptr_t *> malloc(sizeof(intptr_t) * self.size)

  def __dealloc__(self):
    if self._weight: free(self._weight)
    if self._items: free(self._items)

  def __iter__(self):
    for i in range(self.used):
      if self.weighted:
        yield self._items[i], self._weight[i]
      else:
        yield self._items[i]

  def __call__(self, tree, mode, pos, dir=None, size=None,):
    """ For a unweighted search, size is half size of the AABB box.
        For a weighted search, do not give size because it is not used;
         we return 'limit' number of particles instead. """
    if self._weighted:
      return self._execute_weighted(tree, pos)
    else:
      return self._execute_straight(tree, pos, size)

  def execute_straight(self, Tree tree, pos, radius):
    """ find particles within a box at pos +-s.
        return array of array of intp.
        if pos is 1 dim, return array of intp.
        size
    """
    pos = numpy.asarray(pos)
    radius = numpy.asarray(radius)

    outintp = numpy.empty(numpy.broadcast(pos, radius).shape[:-1], dtype='intp')
    used = numpy.empty(numpy.broadcast(pos, radius).shape[:-1], dtype='intp')
    ops = [pos[..., i] for i in range(3)]
    try: ops += [radius[..., i] for i in range(3)]
    except:
      try: ops += [radius[..., 0] for i in range(3)]
      except:
        ops += [radius for i in range(3)]
    ops += [outintp, used]
    iter = numpy.nditer(
       ops, 
       op_flags= [['readonly']] * 6 + [['writeonly']] * 2,
          flags=['buffered', 'external_loop', 'refs_ok'], 
      casting='unsafe', 
      op_dtypes=['f8'] * 6 + ['intp'] * 2)

    cdef npyiter.CIter citer
    cdef size_t size = npyiter.init(&citer, iter)
    cdef double fpos[3], fsize[3]
    cdef int d
    with nogil:
      while size > 0:
        while size > 0:
          for d in range(3):
            fpos[d] = (<double *>citer.data[d])[0]
            fsize[d] = (<double *>citer.data[d+3])[0]
          self.execute_one(tree, fpos, fsize)
          # gonna be before stealing, because self.used is reset!
          (<intptr_t *>citer.data[7])[0] = self.used
          (<void **>citer.data[6])[0] = <void *> (self.steal())
          npyiter.advance(&citer)
          size = size -1
        size = npyiter.next(&citer)

    return construct_arrays(outintp, used, numpy.NPY_INTP)

  def execute_weighted(self, Tree tree, pos):
    """ find nearest self.limit particles around pos.
        return array of (pos.shape, limit).
        when pos has one item, return array with length of (limit)
    """
    pos = numpy.asarray(pos)
    out = numpy.empty(numpy.broadcast(pos, pos).shape[:-1], dtype=[('', ('intp', self.size))])
    ops = [pos[..., i] for i in range(3)]
    ops += [out]

    iter = numpy.nditer(
       ops, 
       op_flags=[['readonly']] * 3 + [ ['writeonly']], 
          flags=['buffered', 'external_loop'], 
      casting='unsafe', 
      op_dtypes=['f8'] * 3 + [out.dtype])

    cdef double fpos[3], fsize[3]
    cdef npyiter.CIter citer
    cdef intptr_t k = 0
    cdef size_t size = npyiter.init(&citer, iter)
    cdef double w
    cdef int d
    cdef node_t node 
    cdef found_last_time = 0
    with nogil:
      while size > 0:
        while size > 0:
          for d in range(3):
            fpos[d] = (<double *>citer.data[d])[0]
          node = tree.get_container(fpos, 0)
          tree.get_node_size(node, fsize)
          while True:
            self.execute_one(tree, fpos, fsize)
            if self.used < self.size: 
              for d in range(3):
                fsize[d] = 1.3 * fsize[d]
              continue
            if self.used == self.size:
              w = sqrt(self._weight[1])
              if w <= fsize[0]:
                break
              else:
                fsize[0] = w
                fsize[1] = w
                fsize[2] = w
                continue
          temp = (<intptr_t *>citer.data[3])
          for k in range(self.used):
            temp[k] = self._items[k + 1]
          npyiter.advance(&citer)
          size = size - 1
        size = npyiter.next(&citer)
    return out

  def execute_raytrace_lcn(self, Tree tree, pos, dir, max):
    """ find largest complete nodes(LCN) that are inside a node that intersects a ray.
        return array of arrays of intp.
        when pos has one item, return array of intp.
 
        an lcn is the topmost level node that has 8 or 0 children.
    """
    pos = numpy.asarray(pos)
    dir = numpy.asarray(dir)
    outintp = numpy.empty(numpy.broadcast(pos, dir).shape[:-1], dtype='intp')
    used = numpy.empty(numpy.broadcast(pos, dir).shape[:-1], dtype='intp')

    ops = [pos[..., i] for i in range(3)]
    ops += [dir[..., i] for i in range(3)]
    ops += [max]
    ops += [outintp, used]

    iter = numpy.nditer(
       ops, 
       op_flags=[['readonly']] * 7 + [ ['writeonly']] * 2, 
          flags=['buffered', 'external_loop', 'zerosize_ok'], 
      casting='unsafe', 
      op_dtypes=['f8'] * 7 + ['intp'] * 2)

    self.AABBkey[0] = 0
    self.AABBkey[1] = -1

    cdef double fpos[3], fdir[3], tE, tL, tLmax
    cdef npyiter.CIter citer
    cdef size_t size = npyiter.init(&citer, iter)
    cdef int d
    with nogil:
      while size > 0:
        while size > 0:
          for d in range(3):
            fpos[d] = (<double *>citer.data[d])[0]
            fdir[d] = (<double *>citer.data[d+3])[0]
          tE = 0
          tL = (<double *>citer.data[6])[0]
          self.raytrace_lcn_r(tree, 0, fpos, fdir, tE, tL)
          # gonna be before stealing, because self.used is reset!
          (<intptr_t *>citer.data[8])[0] = self.used
          (<void **>citer.data[7])[0] = <void *> (self.steal())
          npyiter.advance(&citer)
          size = size - 1
        size = npyiter.next(&citer)

    return construct_arrays(outintp, used, numpy.NPY_INTP)

    
  def execute_raytrace(self, Tree tree, pos, dir, max, node_t root=0):
    """ find particles that are inside a node that intersects a ray.
        return array of arrays of intp.
        when pos has one item, return array of intp.
    """
    pos = numpy.asarray(pos)
    dir = numpy.asarray(dir)
    outintp = numpy.empty(numpy.broadcast(pos, dir).shape[:-1], dtype='intp')
    used = numpy.empty(numpy.broadcast(pos, dir).shape[:-1], dtype='intp')

    ops = [pos[..., i] for i in range(3)]
    ops += [dir[..., i] for i in range(3)]
    ops += [max]
    ops += [outintp, used]

    iter = numpy.nditer(
       ops, 
       op_flags=[['readonly']] * 7 + [ ['writeonly']] * 2, 
          flags=['buffered', 'external_loop', 'zerosize_ok'], 
      casting='unsafe', 
      op_dtypes=['f8'] * 7 + ['intp'] * 2)

    self.AABBkey[0] = 0
    self.AABBkey[1] = -1

    cdef double fpos[3], fdir[3], tE, tL, tLmax
    cdef npyiter.CIter citer
    cdef size_t size = npyiter.init(&citer, iter)
    cdef int d
    with nogil:
      while size > 0:
        while size > 0:
          for d in range(3):
            fpos[d] = (<double *>citer.data[d])[0]
            fdir[d] = (<double *>citer.data[d+3])[0]
          tE = 0
          tL = (<double *>citer.data[6])[0]
          self.raytrace_one_r(tree, root, fpos, fdir, tE, tL)
          # gonna be before stealing, because self.used is reset!
          (<intptr_t *>citer.data[8])[0] = self.used
          (<void **>citer.data[7])[0] = <void *> (self.steal())
          npyiter.advance(&citer)
          size = size - 1
        size = npyiter.next(&citer)

    return construct_arrays(outintp, used, numpy.NPY_INTP)


  property weight:
    def __get__(self):
      cdef numpy.intp_t dims[1]
      if not self._weighted:
        raise AttributeError("attribute weight is unavable for unweighted query object")

      dims[0] = self.used
      arr = numpy.PyArray_SimpleNewFromData(1, dims, numpy.NPY_DOUBLE, self._weight+1)
      numpy.set_array_base(arr, self)
      return arr

  property items:
    def __get__(self):
      cdef numpy.intp_t dims[1]
      dims[0] = self.used
      if self._weighted:
        arr = numpy.PyArray_SimpleNewFromData(1, dims, numpy.NPY_INTP, self._items + 1)
      else:
        arr = numpy.PyArray_SimpleNewFromData(1, dims, numpy.NPY_INTP, self._items)
      numpy.set_array_base(arr, self)
      return arr

  cdef intptr_t * steal(Query self) nogil:
    """ steals the items from the query,
        the query will allocate new internal buffers,
        and reset used to 0, so it's ready for a new execution
    """
    cdef intptr_t * rt = self._items
    if self._weighted:
      self._items = <intptr_t *> malloc(sizeof(intptr_t) * (self.size + 1))
    else:
      self._items = <intptr_t *> malloc(sizeof(intptr_t) * self.size )
    self.used = 0
    return rt

  cdef void raytrace_lcn_r(Query self, Tree tree, node_t node, double p0[3], double dir[3], double tE, double tL) nogil:
    cdef double pos[3], size[3]
    cdef double ttE, ttL
    cdef int i
    cdef int nchildren
    ttE = tE
    ttL = tL
    children = tree.get_node_children(node, &nchildren)
    if nchildren < 8 and nchildren > 0:
      for i in range(nchildren):
        self.raytrace_lcn_r(tree, children[i], p0, dir, tE, tL)
    else:
      tree.get_node_pos(node, pos)
      tree.get_node_size(node, size)
      if LiangBarsky(pos, size, p0, dir, &ttE, &ttL):
        self._add_node_lcn(tree, node)

  cdef void raytrace_one_r(Query self, Tree tree, node_t node, double p0[3], double dir[3], double tE, double tL) nogil:
    cdef double pos[3], size[3]
    cdef double ttE, ttL
    cdef int i
    cdef int nchildren
    ttE = tE
    ttL = tL
    tree.get_node_pos(node, pos)
    tree.get_node_size(node, size)
    if LiangBarsky(pos, size, p0, dir, &ttE, &ttL):
      children = tree.get_node_children(node, &nchildren)
      if nchildren == 0:
        self._add_node_straight(tree, node)
      for i in range(nchildren):
        self.raytrace_one_r(tree, children[i], p0, dir, tE, tL)

  cdef void execute_one(Query self, Tree tree, double pos[3], double size[3]) nogil:
    cdef ipos_t ipos[3]
    cdef int d
    cdef double pos1[3]
    cdef double pos2[3]
    fillingcurve.f2i(tree._scale, pos, ipos)
    fillingcurve.i2fc(ipos, &self.centerkey)
    for d in range(3):
      pos1[d] = pos[d] - size[d]
      pos2[d] = pos[d] + size[d]
    
    fillingcurve.f2i(tree._scale, pos1, ipos)
    fillingcurve.i2fc(ipos, &self.AABBkey[0])
    fillingcurve.f2i(tree._scale, pos2, ipos)
    fillingcurve.i2fc(ipos, &self.AABBkey[1])

    self.used = 0
    self.execute_r(tree, 0)

  cdef void execute_r(Query self, Tree tree, node_t node) nogil:
    cdef int k
    cdef fckey_t key = tree.get_node_key(node)
    cdef int order = tree.get_node_order(node)
    cdef int flag = fillingcurve.heyinAABB(key, order, self.AABBkey)
    cdef int nchildren
    if flag == 0:
      return
    children = tree.get_node_children(node, &nchildren)
    if flag == 2 or nchildren == 0:
      if self._weighted:
        self._add_node_weighted(tree, node)
      else:
        self._add_node_straight(tree, node)
    else:
      for k in range(nchildren):
        self.execute_r(tree, children[k])

  cdef void _add_node_lcn(self, Tree tree, node_t node) nogil:
    while self.size == self.used:
      self.size *= 2
      self._items = <intptr_t *> realloc(self._items, sizeof(intptr_t) * self.size)
    self._items[self.used] = node
    self.used = self.used + 1

  cdef void _add_node_straight(self, Tree tree, node_t node) nogil:
    cdef intptr_t i
    cdef size_t nodenpar = tree.get_node_npar(node)
    cdef intptr_t nodefirst = tree.get_node_first(node)
    while self.size < self.used + nodenpar:
      if self.size < 1048576:
        self.size = self.size * 2
      else:
        if nodenpar > 1048576:
          self.size += nodenpar
        else:
          self.size += self.size / 4

      self._items = <intptr_t *> realloc(self._items, sizeof(intptr_t) * self.size)

    for i in range(nodefirst, 
         nodefirst + nodenpar, 1):
      if fillingcurve.heyinAABB(tree._zkey[i], 0, self.AABBkey):
        self._items[self.used] = i
        self.used = self.used + 1

  cdef void _add_node_weighted(self, Tree tree, node_t node) nogil:
    cdef intptr_t item
    cdef ipos_t id[3]
    cdef double fd[3], weight
    cdef Heap heap
    cdef size_t nodenpar = tree.get_node_npar(node)
    cdef intptr_t nodefirst = tree.get_node_first(node)
    for item in range(nodefirst, 
         nodefirst + nodenpar, 1):
      weight = fillingcurve.key2key2(tree._scale, self.centerkey, tree._zkey[item])

      if self.used < self.size:
        # indexing starts from 1, b/c [0] is scratch for heap
        self._items[self.used+1] = item
        self._weight[self.used+1] = weight
        self.used = self.used + 1
        if self.used == self.size:
          # heapify when full
          makeheap(&heap, self)
          _heapify(&heap)
      else:
        # heap push pop
        if weight < self._weight[1]:
          self._items[1] = item
          self._weight[1] = weight
          makeheap(&heap, self)
          _siftup(&heap, 0)
