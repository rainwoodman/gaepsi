import numpy
cimport numpy
cimport cpython
from ztree cimport Tree
from libc.stdint cimport *
from libc.stdlib cimport malloc, realloc, free
cimport zorder
from zorder cimport zorder_t
cimport npyiter
from libc.math cimport sqrt
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
   
cdef void construct_arrays(numpy.ndarray pointers, numpy.ndarray used, int type):
    """ convert pointers to array of numpy arrays """
    iter = numpy.nditer([pointers, used], 
       op_flags= [['readwrite'], ['readonly']],
          flags=['buffered', 'external_loop', 'refs_ok'], 
      casting='unsafe', 
      op_dtypes=['intp'] * 2)
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
        (<void **>citer.data[0])[0] = <void *>arr
        cpython.Py_INCREF(arr)
        npyiter.advance(&citer)
      size = npyiter.next(&citer)

cdef class Query:

  def __cinit__(self, int limit, int weighted):
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

  def __call__(self, tree, pos, size=None):
    """ For a unweighted search, size is half size of the AABB box.
        For a weighted search, do not give size because it is not used;
         we return 'limit' number of particles instead. """
    if self._weighted:
      return self._execute_weighted(tree, pos)
    else:
      return self._execute_straight(tree, pos, size)

  def _execute_straight(self, Tree tree, pos, s):
    out = numpy.empty(numpy.broadcast(pos, s).shape[:-1], dtype='object')
    used = numpy.empty(numpy.broadcast(pos, s).shape[:-1], dtype='intp')
    outintp = out.view(dtype='intp')
    ops = [pos[..., i] for i in range(3)]
    try: ops += [s[..., i] for i in range(3)]
    except:
      try: ops += [s[..., 0] for i in range(3)]
      except:
        ops += [s for i in range(3)]
    ops += [out, used]
    iter = numpy.nditer(
       ops, 
       op_flags= [['readonly']] * 6 + [['writeonly']] * 2,
          flags=['buffered', 'external_loop', 'refs_ok'], 
      casting='unsafe', 
      op_dtypes=['f8'] * 6 + ['intp'] * 2)

    cdef npyiter.CIter citer
    cdef size_t size = npyiter.init(&citer, iter)
    cdef double fpos[3], fsize[3]
    cdef intptr_t * temp
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

    construct_arrays(outintp, used, numpy.NPY_INTP)
    return out

  def _execute_weighted(self, Tree tree, pos):
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
    cdef intptr_t * temp
    cdef intptr_t k = 0
    cdef size_t size = npyiter.init(&citer, iter)
    cdef double w
    cdef int d
    cdef intptr_t leaf
    cdef found_last_time = 0
    with nogil:
      while size > 0:
        while size > 0:
          for d in range(3):
            fpos[d] = (<double *>citer.data[d])[0]
          leaf = tree.get_container(fpos, 0)
          tree.get_node_size(leaf, fsize)
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

  def _execute_raytrace(self, Tree tree, pos, dir, max):
    out = numpy.empty(numpy.broadcast(pos, dir).shape[:-1], dtype='object')
    used = numpy.empty(numpy.broadcast(pos, dir).shape[:-1], dtype='intp')
    outintp = out.view(dtype='intp')

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
          tL = 0
          tE = (<double *>citer.data[6])[0]
          ###### do something here!
          # gonna be before stealing, because self.used is reset!
          (<intptr_t *>citer.data[8])[0] = self.used
          (<void **>citer.data[7])[0] = <void *> (self.steal())
          npyiter.advance(&citer)
          size = size - 1
        size = npyiter.next(&citer)

    construct_arrays(outintp, used, numpy.NPY_INTP)
    return out

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

  cdef void execute_one(Query self, Tree tree, double pos[3], double size[3]) nogil:
    cdef int32_t ipos[3]
    cdef int d
    cdef double pos1[3]
    cdef double pos2[3]
    tree.digitize.f2i(pos, ipos)
    self.centerkey = zorder.encode(ipos)
    for d in range(3):
      pos1[d] = pos[d] - size[d]
      pos2[d] = pos[d] + size[d]
    
    tree.digitize.f2i(pos1, ipos)
    self.AABBkey[0] = zorder.encode(ipos)
    tree.digitize.f2i(pos2, ipos)
    self.AABBkey[1] = zorder.encode(ipos)

    self.used = 0
    self.execute_r(tree, 0)

  cdef void execute_r(Query self, Tree tree, intptr_t node) nogil:
    cdef int k
    cdef zorder_t key = tree._nodes[node].key
    cdef int order = tree._nodes[node].order
    cdef int flag = zorder.AABBtest(key, order, self.AABBkey)
    if flag == 0:
      return
    if flag == 2 or tree._nodes[node].child_length == 0:
      if self._weighted:
        self._add_node_weighted(tree, node)
      else:
        self._add_node_straight(tree, node)
    else:
      for k in range(tree._nodes[node].child_length):
        self.execute_r(tree, tree._nodes[node].child[k])

  cdef void _add_node_straight(self, Tree tree, intptr_t node) nogil:
    cdef intptr_t i
    while self.size < self.used + tree._nodes[node].npar:
      if self.size < 1048576:
        self.size = self.size * 2
      else:
        if tree._nodes[node].npar > 1048576:
          self.size += tree._nodes[node].npar
        else:
          self.size += self.size / 4

      self._items = <intptr_t *> realloc(self._items, sizeof(intptr_t) * self.size)

    for i in range(tree._nodes[node].first, 
         tree._nodes[node].first + tree._nodes[node].npar, 1):
      if zorder.AABBtest(tree._zkey[i], 0, self.AABBkey):
        self._items[self.used] = i
        self.used = self.used + 1

  cdef void _add_node_weighted(self, Tree tree, intptr_t node) nogil:
    cdef intptr_t item
    cdef int32_t id[3]
    cdef double fd[3], weight
    cdef Heap heap
    for item in range(tree._nodes[node].first, 
         tree._nodes[node].first + tree._nodes[node].npar, 1):
      zorder.diff(self.centerkey, tree._zkey[item], id)
      tree.digitize.i2f0(id, fd)
      weight = fd[0] * fd[0] + fd[1] * fd[1] + fd[2] * fd[2]

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
