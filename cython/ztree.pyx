#cython: embedsignature=True
#cython: cdivision=True
import numpy
cimport cpython
from cpython.ref cimport Py_INCREF
cimport numpy
cimport npyiter
from libc.stdint cimport *
from libc.stdlib cimport malloc, realloc, free
from libc.float cimport FLT_MAX
from libc.limits cimport INT_MAX, INT_MIN
from libc.math cimport fmin
cimport cython
import cython
from warnings import warn
from zorder cimport Zorder

numpy.import_array()

NodeInfoDtype = numpy.dtype([('key', 'i8'), ('order', 'i2'), ('child_length', 'i2'), ('parent', 'i4'), ('first', 'i8'), ('npar', 'i8'), ('child', ('i4', 8))])

cdef inline int insquare(int64_t sqkey, int order, int64_t k2) nogil:
  return 0 == ((sqkey ^ k2) >> (order * 3))

cdef class Result:
  def __cinit__(Result self, int limit = 0):
    if limit > 0:
      self.limit = limit 
      self.size = limit
      self.used = 0
      self._weight = <float *>malloc(sizeof(float) * self.size)
    else:
      self.size = 8
      self.used = 0
    self._buffer = <intptr_t *>malloc(sizeof(intptr_t) * self.size)

  def __dealloc__(self):
    if self._buffer: free(self._buffer)
    if self.limit > 0:
      free(self._weight)

cdef class Tree:

  def __cinit__(self):
    self.size = 1024
    self._buffer = <NodeInfo *>malloc(sizeof(NodeInfo) * self.size)
    self.used = 0

  @cython.boundscheck(False)
  def __init__(self, points, zorder=None, thresh=100):
    """ zorder is an Zorder object.
        points can be given as 'i8' zkeys if zkey is given,
        otherwise points shall be ('f4', 3) and zkey is calculated from points.
    """
    self.thresh = thresh
    if zorder is None:
      if points.shape[-1] != 3:
        raise ValueError('points needs to be of shape = -1, 3 to construct Zorder')
      zorder = Zorder.from_points(points[:, 0], points[:, 1], points[:, 2], bits=21)
    else:
      if points.shape[-1] == 3:
        zkey = zorder(points[:, 0], points[:, 1], points[:, 2])
      else:
        if points.dtype != numpy.dtype('i8'):
          raise ValueError('points need to be either (float, 3) or i8')
        zkey = numpy.array(points, copy=False, order='C')
   
    self.zorder = zorder
    self.zkey = zkey
    self._zkey = <int64_t *> self.zkey.data
    self._zkey_length = self.zkey.shape[0]
    if -1 == self._tree_build():
      raise ValueError("tree build failed. Is the input zkey sorted?")

  def getarray(self):
    """ returns the internal buffer as a recarray, not very useful"""
    cdef numpy.intp_t dims[1]
    dims[0] = self.used * sizeof(NodeInfo)
    arr = numpy.PyArray_SimpleNewFromData(1, dims, numpy.NPY_BYTE, self._buffer)
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
    dims[0] = self._buffer[ind].child_length
    arr = numpy.PyArray_SimpleNewFromData(1, dims, numpy.NPY_INT, self._buffer[ind].child)
    numpy.set_array_base(arr, self)
    rt = dict(key=self._buffer[ind].key, 
            order=self._buffer[ind].order,
           parent=self._buffer[ind].parent, 
            index=ind,
         children=arr)

    if self._buffer[ind].child_length == 0:
       rt.update(dict(first=self._buffer[ind].first, 
                      last=self._buffer[ind].first + self._buffer[ind].npar))
    else:
       rt.update(dict(first=0, last=-1))
    return rt
    
  @cython.boundscheck(False)
  cdef void query_neighbours_one(Tree self, Result result, float pos[3]) nogil:
     cdef intptr_t j
     cdef int32_t r
     cdef int64_t key
     cdef intptr_t tmp
     cdef int32_t min[3], max[3], center[3]
     key = self.zorder.encode_float(pos)
     r = self.query_neighbours_estimate_radius(key, result.limit)
     cdef float max_weight = 0
     cdef int iteration = 0
     while iteration < 4:
       result.truncate()
       self.zorder.BBint(pos, r / self.zorder._norm[0], center, min, max)
       self.query_box_one(result, min, max, center)
       r = (r << 1)
       iteration = iteration + 1
       if result.used < result.limit:
         continue
       if max_weight == result._weight[result.limit -1]: 
         break
       else:
         max_weight = result._weight[result.limit -1]
     if iteration == 4:
       with gil:
         warn('query neighbour one failed, returning %d /%d', result.used, result.limit)
       for j in range(result.used, result.limit):
         result._buffer[j] = -1

  @cython.boundscheck(False)
  def query_neighbours(Tree self, x, y, z, int32_t count):
    cdef intptr_t i, d, j, k
    cdef int64_t [:] queryzkey
    cdef float pos[3]

    out = numpy.empty_like(x, dtype=[('data', (numpy.intp, count))])
    iter = numpy.nditer([x, y, z, out], op_flags=[['readonly'], ['readonly'], ['readonly'], ['writeonly']], flags=['buffered', 'external_loop'], casting='unsafe', op_dtypes=['f4', 'f4', 'f4', out.dtype])

    cdef npyiter.NpyIter * citer = npyiter.GetNpyIter(iter)
    cdef npyiter.IterNextFunc next = npyiter.GetIterNext(citer, NULL)
    cdef char ** data = npyiter.GetDataPtrArray(citer)
    cdef numpy.npy_intp *strides = npyiter.GetInnerStrideArray(citer)
    cdef numpy.npy_intp *size_ptr = npyiter.GetInnerLoopSizePtr(citer)
    cdef intptr_t iop, size
    cdef intptr_t total = 0
    cdef Result result = Result(count)

    with nogil:
     while True:
      size = size_ptr[0]
      total += size
      while size > 0:
        for d in range(3):
          pos[d] = (<float*>data[d])[0]
        result.truncate()
        self.query_neighbours_one(result, pos)
        for i in range(count):
          (<intptr_t*>data[3])[i] = result._buffer[i]

        for iop in range(4):
          data[iop] += strides[iop]
        size = size - 1
      i = next(citer)
      if i == 0: break

    out = out.view(dtype=(numpy.intp, count))
    return out
  
  @cython.boundscheck(False)
  cdef int32_t query_neighbours_estimate_radius(Tree self, int64_t ckey, int count) nogil:
    cdef intptr_t this, child, next
    cdef float rt = 0, tmp = 0
    this = 0
    while this != -1 and self._buffer[this].child_length > 0:
      next = this
      for i in range(self._buffer[this].child_length):
        child = self._buffer[this].child[i]
        if insquare(self._buffer[child].key, self._buffer[child].order, ckey):
          next = child
          break
      if next == this: break
      else:
        if self._buffer[next].npar < count: break
        this = next
        continue

    while this != -1 and self._buffer[this].npar < count:
      this = self._buffer[this].parent

    if this == -1: this = 0
    return ((1 << self._buffer[this].order) - 1)
    

  @cython.boundscheck(False)
  def query_box(Tree self, x, y, z, radius, int limit = 0):
    oshape = numpy.asarray(x).shape
    radius = numpy.atleast_1d(radius)
    cdef int32_t min[3]
    cdef int32_t max[3]
    cdef int32_t center[3]
    cdef Result result
    cdef intptr_t i
    cdef float pos[3]
    cdef numpy.ndarray ret 

    ret = numpy.empty_like(x, dtype=numpy.dtype('object'))
    iter = numpy.nditer([x, y, z, radius, ret], op_flags=[['readonly'], ['readonly'], ['readonly'], ['readonly'], ['readwrite']], flags=['buffered', 'refs_ok', 'external_loop'], casting='unsafe', op_dtypes=['f4', 'f4', 'f4', 'f4', numpy.dtype('object')])

    cdef npyiter.NpyIter * citer = npyiter.GetNpyIter(iter)
    cdef npyiter.IterNextFunc next = npyiter.GetIterNext(citer, NULL)
    cdef char ** data = npyiter.GetDataPtrArray(citer)
    cdef numpy.npy_intp *strides = npyiter.GetInnerStrideArray(citer)
    cdef numpy.npy_intp *size_ptr = npyiter.GetInnerLoopSizePtr(citer)
    cdef intptr_t iop, size
    cdef object a
    # this cannot be done with no gil.

    while True:
      size = size_ptr[0]
      while size > 0:
        for d in range(3):
          pos[d] = (<float*>data[d])[0]
        r = (<float*> data[3])[0]
        self.zorder.BBint(pos, r, center, min, max)
        result = Result(limit)
        self.query_box_one(result, min, max, center)
        a = (result.harvest())
        (<void**>data[4])[0] = <void*> a
        Py_INCREF(a)
        for iop in range(5):
          data[iop] += strides[iop]
        size = size - 1
      i = next(citer)
      if i == 0: break

    return ret

  @cython.boundscheck(False)
  def query_box_wh(Tree self, x, y, z, x1, y1, z1, int limit = 0):
    oshape = numpy.asarray(x).shape
    cdef int32_t min[3]
    cdef int32_t max[3]
    cdef int32_t center[3]
    cdef Result result
    cdef intptr_t i
    cdef float pos[3]
    cdef numpy.ndarray ret 

    ret = numpy.empty_like(x, dtype=numpy.dtype('object'))
    iter = numpy.nditer([x, y, z, x1, y1, z1, ret], 
      op_flags=[['readonly'], ['readonly'], ['readonly'], ['readonly'], ['readonly'], ['readonly'], ['readwrite']], 
      flags=['buffered', 'refs_ok', 'external_loop'], 
      casting='unsafe', op_dtypes=['f4', 'f4', 'f4', 'f4', 'f4', 'f4', numpy.dtype('object')])

    cdef npyiter.NpyIter * citer = npyiter.GetNpyIter(iter)
    cdef npyiter.IterNextFunc next = npyiter.GetIterNext(citer, NULL)
    cdef char ** data = npyiter.GetDataPtrArray(citer)
    cdef numpy.npy_intp *strides = npyiter.GetInnerStrideArray(citer)
    cdef numpy.npy_intp *size_ptr = npyiter.GetInnerLoopSizePtr(citer)
    cdef intptr_t iop, size
    cdef object a
    # this cannot be done with no gil.
    while True:
      size = size_ptr[0]
      while size > 0:
        for d in range(3):
          pos[d] = (<float*>data[d])[0]
        self.zorder.float_to_int(pos, min)
        for d in range(3):
          pos[d] = (<float*>data[3+d])[0]
        self.zorder.float_to_int(pos, max)

        for d in range(3):
          pos[d] = (<float*>data[3+d])[0] + (<float*>data[d])[0] 
          pos[d] *= 0.5
        self.zorder.float_to_int(pos, center)

        result = Result(limit)
        self.query_box_one(result, min, max, center)
        a = (result.harvest())
        (<void**>data[6])[0] = <void*> a
        Py_INCREF(a)
        for iop in range(7):
          data[iop] += strides[iop]
        size = size - 1
      i = next(citer)
      if i == 0: break

    return ret

  @cython.boundscheck(False)
  cdef void __add_node(Tree self, Result result, int32_t min[3], int32_t max[3], int32_t center[3], intptr_t node) nogil:
    """ add the pars in a node into result, use the distance if result.limit > 0"""
    cdef float dx, x
    cdef int64_t key
    cdef intptr_t i, d, j
    cdef int32_t ipos[3]
    cdef int good 
    if result.limit == 0:
      # no need to check the distance, directly add all pars to the result
      result.append(self._buffer[node].first, self._buffer[node].npar)
      return

    for i in range(self._buffer[node].first, self._buffer[node].first + self._buffer[node].npar):
        self.zorder.decode(self._zkey[i], ipos)

        good = 1
        for d in range(3):
          if ipos[d] > max[d] or ipos[d] < min[d]:
            good = 0
            break

        if good: 
          if result.limit == 0:
            result.append_one(i)
          else: 
            x = self.zorder.dist2(ipos, center)
            result.append_one_with_weight(i, x)
    
  cdef int __goodness(Tree self, intptr_t node, int32_t min[3], int32_t max[3]) nogil:
    """ -1 is not overlapping at all, 0 is partially overlapping, 1 is fully inside """
    cdef int32_t cmin[3]
    cdef intptr_t i, j, d
    cdef int32_t box = (1 << self._buffer[node].order) - 1

    self.zorder.decode(self._buffer[node].key, cmin)

    for d in range(3):
      if cmin[d] > max[d] or cmin[d] + box < min[d]:
        # do not overlap at all
        return -1

    for d in range(3):
      if cmin[d] < min[d] or cmin[d] + box> max[d]:
        return 0
    return 1
     
  @cython.boundscheck(False)
  cdef void query_box_one(Tree self, Result result, int32_t min[3], int32_t max[3], int32_t center[3]) nogil:
    self.__query_box_one_from(result, min, max, center, 0)

  cdef void __query_box_one_from(Tree self, Result result, int32_t min[3], int32_t max[3], int32_t center[3], intptr_t root) nogil:
    cdef int verygood = self.__goodness(root, min, max)
    if verygood == -1: return
    if self._buffer[root].child_length == 0 or verygood == 1:
      # has no children or fully inside, open the node, pump in
      self.__add_node(result, min, max, center, root)
    else:
      # loop over the children
      for i in range(self._buffer[root].child_length):
        self.__query_box_one_from(result, min, max, center, self._buffer[root].child[i])

  @cython.boundscheck(False)
  cdef int _tree_build(Tree self) nogil:
      cdef intptr_t j = 0
      cdef intptr_t i = 0
      cdef intptr_t step = 0
      self.used = 1;
      self._buffer[0].key = 0
      self._buffer[0].first = 0
      self._buffer[0].npar = 0
      self._buffer[0].order = self.zorder.bits
      self._buffer[0].parent = -1
      self._buffer[0].child_length = 0
      while i < self._zkey_length:
        while not insquare(self._buffer[j].key, self._buffer[j].order, self._zkey[i]):
          # close the nodes by filling in the npar, because we already scanned over
          # all particles in these nodes.
          self._buffer[j].npar = i - self._buffer[j].first
          j = self._buffer[j].parent
          # because we are on a morton key ordered list, no need to deccent into children 
        # NOTE: will never go beyond 8 children per node, 
        # for the child_length > 0 branch is called less than 8 times on the parent, 
        # the point fails in_square of the current node
        if self._buffer[j].child_length > 0:
          # already not a leaf, create new child
          j = self._create_child(i, j) 
        elif (self._buffer[j].npar >= self.thresh and self._buffer[j].order > 0):
          # too many points in the leaf, split it
          # NOTE: i is rewinded, because now some of the particles are no longer
          # in the new node.
          i = self._buffer[j].first
          j = self._create_child(i, j) 
        else:
          # put the particle into the leaf.
          self._buffer[j].npar = self._buffer[j].npar + 1
        if j == -1: return -1
        # now we try to fast forword to the first particle that is not in the current node
        step = self.thresh
        if i + step < self._zkey_length:
          while not insquare(self._buffer[j].key, self._buffer[j].order, self._zkey[i + step]):
            step >>= 1
          if step > 0:
            self._buffer[j].npar = self._buffer[j].npar + step
            i = i + step
        i = i + 1
      # now close the remaining open nodes
      while j >= 0:
        self._buffer[j].npar = i - self._buffer[j].first
        j = self._buffer[j].parent
        
        
  @cython.boundscheck(False)
  cdef intptr_t _create_child(self, intptr_t first_par, intptr_t parent) nogil:
    # creates a child of parent from first_par, returns the new child */
    self._buffer[self.used].first = first_par
    self._buffer[self.used].npar = 1
    self._buffer[self.used].parent = parent
    self._buffer[self.used].child_length = 0
    self._buffer[self.used].order = self._buffer[parent].order - 1
    #/* the lower bits of a sqkey is cleared off but I don't think it is necessary */
    self._buffer[self.used].key = (self._zkey[first_par] >> (self._buffer[self.used].order * 3)) << (self._buffer[self.used].order * 3)
    self._buffer[parent].child[self._buffer[parent].child_length] = self.used
    self._buffer[parent].child_length = self._buffer[parent].child_length + 1
    if self._buffer[parent].child_length > 8:
      return -1
    cdef intptr_t rt = self.used
    self.used = self.used + 1
    if self.used == self.size:
      self._grow()
    return rt

  def __dealloc__(self):
    free(self._buffer)


