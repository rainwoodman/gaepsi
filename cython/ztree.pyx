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

numpy.import_array()

NodeInfoDtype = numpy.dtype([('key', 'i8'), ('order', 'i2'), ('child_length', 'i2'), ('parent', 'i4'), ('first', 'i8'), ('npar', 'i8'), ('child', ('i4', 8))])

#def _(func, args):
#  signature = []
#  for x in args:
#    if isinstance(x, (numpy.ndarray, numpy.generic)):
#      if x.dtype is numpy.dtype('f4'):
#        signature += ['float[:]']
#      if x.dtype is numpy.dtype('f8'):
#        signature += ['double[:]']
#      if x.dtype is numpy.dtype('i2'):
#        signature += ['short[:]']
#      if x.dtype is numpy.dtype('i4'):
#        signature += ['int[:]']
#      if x.dtype is numpy.dtype('i8'):
#        signature += ['long[:]']
#    else:
#      signature += [cython.typeof(x)]
#  try:
#    return func.__signatures__[', '.join(signature)]
#  except:
#    print signature, func.__signatures__
#    raise

cdef inline int insquare(int64_t sqkey, int order, int64_t k2) nogil:
  return 0 == ((sqkey ^ k2) >> (order * 3))

cdef inline float dist2(int64_t key1, int64_t key2, float norm[3], int bits) nogil:
    cdef int64_t key = key1 ^ key2
    cdef float x, y, z
    cdef int j
    cdef int32_t ix = 0, iy = 0, iz = 0
    for j in range(0, bits) :
      iz = iz + ((key & 1) << j)
      key = key >>1
      iy = iy + ((key & 1) << j)
      key = key >>1
      ix = ix + ((key & 1) << j)
      key = key >>1
    x = ix / norm[0]
    y = iy / norm[1]
    z = iz / norm[2]
    return x * x + y * y + z * z

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

  cpdef harvest(Result self):
    cdef cython.view.array array
    array = <intptr_t[:self.used]> self._buffer
    array.callback_free_data = free
    self._buffer = NULL
    return numpy.asarray(array)

  @cython.boundscheck(False)
  cdef void _grow(Result self) nogil:
    if self.size < 1024576:
      self.size *= 2
    else:
      self.size += 1024576
    self._buffer = <intptr_t *>realloc(self._buffer, sizeof(intptr_t) * self.size)
    if self.limit > 0:
      self._weight = <float *>realloc(self._buffer, sizeof(float) * self.size)

  cdef void truncate(Result self) nogil:
    self.used = 0

  @cython.boundscheck(False)
  cdef void append_one(Result self, intptr_t i) nogil:
    if self.size - self.used <= 1:
      self._grow()
    self._buffer[self.used] = i
    self.used = self.used + 1

  @cython.boundscheck(False)
  cdef void append_one_with_weight(Result self, intptr_t i, float weight) nogil:
    cdef intptr_t k
    if self.used == self.limit:
      if weight >= self._weight[self.used - 1]: return
    else:
      self.used = self.used + 1
    k = self.used - 1
    while k > 0 and weight < self._weight[k - 1]:
      self._weight[k] = self._weight[k - 1]
      self._buffer[k] = self._buffer[k - 1]
      k = k - 1
    self._weight[k] = weight
    self._buffer[k] = i

  @cython.boundscheck(False)
  cdef void append(Result self, intptr_t start, intptr_t length) nogil:
    cdef intptr_t i
    while self.size - self.used <= length:
      self._grow()
    for i in range(start, start + length):
      self._buffer[self.used] = i
      self.used = self.used + 1

  def __dealloc__(self):
    if self._buffer: free(self._buffer)
    if self.limit > 0:
      free(self._weight)

cdef class Scale(object):
  """Scale scales x,y,z to 0 ~ (1<<bits) - 1 """
  def __cinit__(self):
    self._min = <float*> malloc(sizeof(float) * 3)
    self._norm = <float*> malloc(sizeof(float) * 3)
    self.min = numpy.asarray(<float[:3]>self._min)
    self.norm = numpy.asarray(<float[:3]>self._norm)
  def __dealloc(self):
    self.min = None
    self.norm = None
    free(self._min)
    free(self._norm)
  def __init__(self, x=None, y=None, z=None, min=None, norm=None, bits=1):
    if min is not None:
      self.min[:] = numpy.asarray(min, dtype=numpy.float32)
      self.norm[:] = numpy.asarray(norm, dtype=numpy.float32)
    elif x is not None:
      self.min[:] = numpy.array([x.min(), y.min(), z.min()], dtype=numpy.float32)
      self.norm[:] = numpy.array([
         1 / (x.max() - self.min[0]),
         1 / (y.max() - self.min[1]),
         1 / (z.max() - self.min[2])], dtype=numpy.float32)
      self.norm *= ((1 << bits) - 1)
    else:
      self.min[:] = numpy.array([0., 0., 0.])
      self.norm[:] = numpy.array([1., 1., 1.])
      self.norm *= ((1 << bits) - 1)
    self.bits = bits

  def __call__(self, x, y, z, ix=None, iy=None, iz=None):
    x = numpy.asarray(x)
    y = numpy.asarray(y)
    z = numpy.asarray(z)

    if ix is None:
      ix = numpy.empty(shape=x.shape, dtype='i4')
    if iy is None:
      iy = numpy.empty(shape=y.shape, dtype='i4')
    if iz is None:
      iz = numpy.empty(shape=z.shape, dtype='i4')

    ix[...] = (x - self.min[0]) * self.norm[0]
    iy[...] = (y - self.min[1]) * self.norm[1]
    iz[...] = (z - self.min[2]) * self.norm[2]

    return ix, iy, iz

  def __str__(self):
    return str(dict(min=self.min, norm=self.norm))

  cdef void BBint(Scale self, float pos[3], float r, int32_t center[3], int32_t min[3], int32_t max[3]) nogil:
    cdef float rf, f
    for d in range(3):
      center[d] = <int32_t> ((pos[d] - self._min[d] ) * self._norm[d])
      rf = r * self._norm[d]
      f = center[d] - rf
      if f > INT_MAX: min[d] = INT_MAX
      elif f < INT_MIN: min[d] = INT_MIN
      else: min[d] = <int32_t>f

      f = center[d] + rf
      if f > INT_MAX: max[d] = INT_MAX
      elif f < INT_MIN: max[d] = INT_MIN
      else: max[d] = <int32_t>f
    
  cdef void from_float(Scale self, float pos[3], int32_t point[3]) nogil:
    cdef int d
    for d in range(3):
      point[d] = <int32_t> ((pos[d] - self._min[d]) * self._norm[d])

  cdef float dist2(Scale self, int32_t center[3], int32_t point[3]) nogil:
    cdef float x, dx
    cdef int d
    x = 0
    for d in range(3):
       dx = (point[d] - center[d]) / self._norm[d]
       x += dx * dx
    return x

  cdef void decode(Scale self, int64_t key, int32_t point[3]) nogil:
    cdef int j
    point[0] = 0
    point[1] = 0
    point[2] = 0
    for j in range(0, self.bits) :
      point[2] += ((key & 1) << j)
      key >>= 1
      point[1] += ((key & 1) << j)
      key >>= 1
      point[0] += ((key & 1) << j)
      key >>= 1
  cdef void decode_float(Scale self, int64_t key, float pos[3]) nogil:
    cdef int32_t point[3]
    cdef int d
    self.decode(key, point)
    for d in range(3):
      pos[d] = point[d] / self._norm[d] + self._min[d]
    
  cdef int64_t encode(Scale self, int32_t point[3]) nogil:
    cdef int64_t key = 0
    cdef int j
    for j in range(self.bits-1, -1, -1) :
      key = key << 1
      key = key + ((point[0] >> j) & 1)
      key = key << 1
      key = key + ((point[1] >> j) & 1)
      key = key << 1
      key = key + ((point[2] >> j) & 1)
    return key

  cdef int64_t encode_float (Scale self, float pos[3]) nogil:
    cdef int32_t point[3]
    cdef int d
    for d in range(3):
      point[d] = <int32_t> ((pos[d] - self._min[d]) * self._norm[d])
    return self.encode(point)

cdef class Tree(object):

  def __cinit__(self):
    self.size = 1024
    self._buffer = <NodeInfo *>malloc(sizeof(NodeInfo) * self.size)
    self.used = 0

  @cython.boundscheck(False)
  def __init__(self, xyz=None, zorder=None, scale=None, thresh=100):
    """ if zorder = None, xyz = (x, y, z) will be sorted in place
        if zorder is not None, assume the input is already ordered
        scale it transforms the position to [0, 1]
    """
    self.thresh = thresh
    if zorder is None and xyz is not None:
      x, y, z = xyz
      self.zorder = numpy.empty(shape=x.shape[0], dtype=numpy.intp)
      self._zorder = self.zorder
      bits = self.zorder.dtype.itemsize * 8 / 3
      self.scale = Scale(x, y, z, bits=bits)
      _zorder(x, y, z, out=self.zorder, scale=self.scale)
      ind = numpy.asarray(self.zorder).argsort()
      tmp = x[ind]
      x[:] = tmp
      tmp = y[ind]
      y[:] = tmp
      tmp = z[ind]
      z[:] = tmp
      self.zorder[:] = self.zorder[ind]
    elif zorder is not None and xyz is None and scale is not None:
      self.zorder = zorder
      self._zorder = zorder
      self.scale = scale
    else:
      raise ValueError("give xyz, or zorder and scale")

    if -1 == self._tree_build():
      raise ValueError("tree build failed. Is the input zorder sorted?")

  @cython.boundscheck(False)
  cdef void _grow(self) nogil:
    if self.size < 1024576 * 16:
      self.size *= 2
    else:
      self.size += 1024576 * 16
    self._buffer = <NodeInfo * >realloc(self._buffer, sizeof(NodeInfo) * self.size)

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
     key = self.scale.encode_float(pos)
     r = self.query_neighbours_estimate_radius(key, result.limit)
     cdef float max_weight = 0
     cdef int iteration = 0
     while iteration < 4:
       result.truncate()
       self.scale.BBint(pos, r / self.scale._norm[0], center, min, max)
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
    cdef int64_t [:] queryzorder
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
        self.scale.BBint(pos, r, center, min, max)
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
        self.scale.from_float(pos, min)
        for d in range(3):
          pos[d] = (<float*>data[3+d])[0]
        self.scale.from_float(pos, max)

        for d in range(3):
          pos[d] = (<float*>data[3+d])[0] + (<float*>data[d])[0] 
          pos[d] *= 0.5
        self.scale.from_float(pos, center)

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
        self.scale.decode(self._zorder[i], ipos)

        good = 1
        for d in range(3):
          if ipos[d] > max[d] or ipos[d] < min[d]:
            good = 0
            break

        if good: 
          if result.limit == 0:
            result.append_one(i)
          else: 
            x = self.scale.dist2(ipos, center)
            result.append_one_with_weight(i, x)
    
  cdef int __goodness(Tree self, intptr_t node, int32_t min[3], int32_t max[3]) nogil:
    """ -1 is not overlapping at all, 0 is partially overlapping, 1 is fully inside """
    cdef int32_t cmin[3]
    cdef intptr_t i, j, d
    cdef int32_t box = (1 << self._buffer[node].order) - 1

    self.scale.decode(self._buffer[node].key, cmin)

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
      self._buffer[0].order = self.scale.bits
      self._buffer[0].parent = -1
      self._buffer[0].child_length = 0
      while i < self._zorder.shape[0]:
        while not insquare(self._buffer[j].key, self._buffer[j].order, self._zorder[i]):
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
        if i + step < self._zorder.shape[0]:
          while not insquare(self._buffer[j].key, self._buffer[j].order, self._zorder[i + step]):
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
    self._buffer[self.used].key = (self._zorder[first_par] >> (self._buffer[self.used].order * 3)) << (self._buffer[self.used].order * 3)
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


def _zorder(x, y, z, out=None, scale=None, bits=None):
  """ if bits is none, use scale.bits. if scale is also none, infer from out """
  if out is None:
    out = numpy.empty(shape=x.shape, dtype='i8')

  if bits is None:
    if scale is not None:
      bits = scale.bits
    else:
      bits = out.dtype.itemsize * 8 / 3
   
  if scale is None:
    scale = Scale(x, y, z, bits=bits)

  if not ( out.shape[0] == x.shape[0] and 
     out.shape[0] == y.shape[0] and
     out.shape[0] == z.shape[0] ):
    raise ValueError('shape mismatch')

  _fused_zorder(out, x, y, z, scale)
     
  return out
zorder = _zorder

@cython.boundscheck(False)
cdef void _fused_zorder(
   int64_t[:] out,
   float[:] x,
   float[:] y,
   float[:] z,
   Scale scale
):
    cdef intptr_t i
    cdef float pos[3]
    for i in range(x.shape[0]):
      pos[0] = x[i]
      pos[1] = y[i]
      pos[2] = z[i]
      out[i] = scale.encode_float(pos)

def zorder_inverse(zorder, scale, x=None, y=None, z=None):
  if x is None:
    x = numpy.empty(zorder.shape, dtype=numpy.float32)
  if y is None:
    y = numpy.empty(zorder.shape, dtype=numpy.float32)
  if z is None:
    z = numpy.empty(zorder.shape, dtype=numpy.float32)

  if not ( zorder.shape[0] == x.shape[0] and 
     zorder.shape[0] == y.shape[0] and
     zorder.shape[0] == z.shape[0] ):
    raise ValueError('shape mismatch')

  _fused_zorder_inverse(zorder, x, y, z, scale)
  return x, y, z

@cython.boundscheck(False)
cdef void _fused_zorder_inverse(
   int64_t [:] zorder,
   float[:] x,
   float[:] y,
   float[:] z,
   Scale scale,
) nogil:
    cdef intptr_t i, j
    cdef int32_t ix, iy, iz
    cdef int64_t key
    cdef float pos[3]
    for i in range(x.shape[0]):
      scale.decode_float(zorder[i], pos)
      x[i] = pos[0]
      y[i] = pos[1]
      z[i] = pos[2]
