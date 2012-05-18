#cython: embedsignature=True
#cython: cdivision=True
import numpy
cimport cpython
cimport numpy
from libc.stdint cimport *
from libc.stdlib cimport malloc, realloc, free
from libc.float cimport FLT_MAX
from libc.limits cimport INT_MAX, INT_MIN

cimport cython
import cython

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

cdef int insquare(int64_t sqkey, int order, int64_t k2) nogil:
  return 0 == ((sqkey ^ k2) >> (order * 3))

cdef float dist2(int64_t key1, int64_t key2, float norm[3], int bits) nogil:
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

cdef class Result(object):
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
    oshape = x.shape
    x = numpy.atleast_1d(x)
    y = numpy.atleast_1d(y)
    z = numpy.atleast_1d(z)
    
    if ix is None:
      ix = numpy.empty(shape=x.shape, dtype='i4')
    if iy is None:
      iy = numpy.empty(shape=y.shape, dtype='i4')
    if iz is None:
      iz = numpy.empty(shape=z.shape, dtype='i4')

    ix[:] = (x - self.min[0]) * self.norm[0]
    iy[:] = (y - self.min[1]) * self.norm[1]
    iz[:] = (z - self.min[2]) * self.norm[2]

    ix.shape = oshape
    iy.shape = oshape
    iz.shape = oshape
    return ix, iy, iz

  def __str__(self):
    return str(dict(min=self.min, norm=self.norm))

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
  def query_neighbours(Tree self, x, y, z, int32_t count):
    cdef intptr_t [:] cand
    cdef intptr_t i, d, j, k, used
    cdef int64_t [:] queryzorder
    cdef float normmax, norm[3], smin[3], x2, r, rf, f
    cdef int32_t min[3], max[3], center[3]
    cdef numpy.ndarray[numpy.intp_t, ndim=2] out
    oshape = numpy.array(x).shape

    normmax = self.scale.norm[0]
    if self.scale.norm[1] > normmax:
      normmax = self.scale.norm[1]
    if self.scale.norm[2] > normmax:
      normmax = self.scale.norm[2]
    for i in range(3):
      norm[i] = self.scale.norm[i]
      smin[i] = self.scale.min[i]
    x = numpy.atleast_1d(x)
    y = numpy.atleast_1d(y)
    z = numpy.atleast_1d(z)
    queryzorder = _zorder(x, y, z, scale=self.scale)
    
    out = numpy.empty(shape=(x.shape[0], count), dtype=numpy.intp)

    for i in range(x.shape[0]):
      r = self.__query_neighbours_estimate_radius(queryzorder[i], normmax, count)
      while True:
        center[0] = (x[i] - smin[0]) * norm[0]
        center[1] = (y[i] - smin[1]) * norm[1]
        center[2] = (z[i] - smin[2]) * norm[2]
        for d in range(3):
          rf = r * norm[d]
          f = center[d] - rf
          if f > INT_MAX: min[d] = INT_MAX
          elif f < INT_MIN: min[d] = INT_MIN
          else: min[d] = <int32_t>f

          f = center[d] + rf
          if f > INT_MAX: max[d] = INT_MAX
          elif f < INT_MIN: max[d] = INT_MIN
          else: max[d] = <int32_t>f

        result = Result(count)
        self.__query_box_one(result, min, max, center, norm, 0)
        if result.used == count: break
        r *= 2
        
      for j in range(count):
        out[i, j] = result._buffer[j]

    return out.reshape(list(oshape) + [count])
  
  @cython.boundscheck(False)
  cdef float __query_neighbours_estimate_radius(Tree self, int64_t ckey, float norm, int count) nogil:
    cdef intptr_t this, child
    cdef float rt = 0, tmp = 0
    this = 0
    while self._buffer[this].child_length > 0:
      for i in range(self._buffer[this].child_length):
        child = self._buffer[this].child[i]
        if insquare(self._buffer[child].key, self._buffer[child].order, ckey):
          this = child
          break
      if self._buffer[this].npar < count:
        this = self._buffer[this].parent
        break
    if this == -1: this = 0
    return ((1 << self._buffer[this].order) - 1) / norm
    

  @cython.boundscheck(False)
  def query_box(Tree self, x, y, z, radius, int limit = 0):
    ix0, iy0, iz0 = self.scale(x, y, z)
    oshape = numpy.asarray(x).shape
    cdef int32_t [:] ix = numpy.atleast_1d(ix0)
    cdef int32_t [:] iy = numpy.atleast_1d(iy0)
    cdef int32_t [:] iz = numpy.atleast_1d(iz0)
    radius = numpy.atleast_1d(radius)
    cdef int32_t min[3]
    cdef int32_t max[3]
    cdef int32_t center[3]
    cdef float rf
    cdef Result result
    cdef intptr_t i, d
    cdef float f
    cdef float r
    cdef float norm[3]
    cdef numpy.ndarray[cython.object, ndim=1] ret = numpy.zeros(ix.shape[0], dtype=numpy.dtype('object'))

    for d in range(3):
      norm[d] = self.scale.norm[d]

    for i in range(ix.shape[0]):
      center[0] = ix[i]
      center[1] = iy[i]
      center[2] = iz[i]
      if radius.shape[0] > 1: r = radius[i]
      else: r = radius[0]
      for d in range(3):
        rf = r * norm[d]
        f = center[d] - rf
        if f > INT_MAX: min[d] = INT_MAX
        elif f < INT_MIN: min[d] = INT_MIN
        else: min[d] = <int32_t>f

        f = center[d] + rf
        if f > INT_MAX: max[d] = INT_MAX
        elif f < INT_MIN: max[d] = INT_MIN
        else: max[d] = <int32_t>f

      result = Result(limit)
      self.__query_box_one(result, min, max, center, norm, 0)
      ret[i] = result.harvest()

    return ret.reshape(oshape)

  @cython.boundscheck(False)
  cdef void __add_node(Tree self, Result result, int32_t min[3], int32_t max[3], int32_t center[3], float norm[3], intptr_t node) nogil:
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
        key = self._zorder[i]
        ipos[0] = 0
        ipos[1] = 0
        ipos[2] = 0
        for j in range(0, self.scale.bits) :
          ipos[2] += ((key & 1) << j)
          key >>= 1
          ipos[1] += ((key & 1) << j)
          key >>= 1
          ipos[0] += ((key & 1) << j)
          key >>= 1
        good = 1
        for d in range(3):
          if ipos[d] > max[d] or ipos[d] < min[d]:
            good = 0
            break
        if good: 
          if result.limit == 0:
            result.append_one(i)
          else: 
            x = 0
            for d in range(3):
               dx = (ipos[d] - center[d]) * norm[d]
               x += dx * dx
            result.append_one_with_weight(i, x)
    
  cdef int __goodness(Tree self, intptr_t node, int32_t min[3], int32_t max[3]) nogil:
    """ -1 is not overlapping at all, 0 is partially overlapping, 1 is fully inside """
    cdef int32_t cmin[3]
    cdef intptr_t i, j, d
    cdef int32_t box = (1 << self._buffer[node].order) - 1
    cdef int64_t key = self._buffer[node].key
    cmin[0] = 0
    cmin[1] = 0
    cmin[2] = 0
    for j in range(0, self.scale.bits) :
      cmin[2] += ((key & 1) << j)
      key >>= 1
      cmin[1] += ((key & 1) << j)
      key >>= 1
      cmin[0] += ((key & 1) << j)
      key >>= 1

    for d in range(3):
      if cmin[d] > max[d] or cmin[d] + box < min[d]:
        # do not overlap at all
        return -1

    for d in range(3):
      if cmin[d] < min[d] or cmin[d] + box> max[d]:
        return 0
    return 1
     
  @cython.boundscheck(False)
  cdef void __query_box_one(Tree self, Result result, int32_t min[3], int32_t max[3], int32_t center[3], float norm[3], intptr_t root) nogil:
    cdef int verygood = self.__goodness(root, min, max)
    if verygood == -1: return
    if self._buffer[root].child_length == 0 or verygood == 1:
      # has no children or fully inside, open the node, pump in
      self.__add_node(result, min, max, center, norm, root)
    else:
      # loop over the children
      for i in range(self._buffer[root].child_length):
        self.__query_box_one(result, min, max, center, norm, self._buffer[root].child[i])

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
    cdef intptr_t i, j
    cdef int32_t ix, iy, iz
    cdef int64_t key

    for i in range(x.shape[0]):
      ix = <int32_t> (scale._norm[0] * (x[i] - scale._min[0]))
      iy = <int32_t> (scale._norm[1] * (y[i] - scale._min[1]))
      iz = <int32_t> (scale._norm[2] * (z[i] - scale._min[2]))
      key = 0
      for j in range(scale.bits-1, -1, -1) :
        key = key << 1
        key = key + ((ix >> j) & 1)
        key = key << 1
        key = key + ((iy >> j) & 1)
        key = key << 1
        key = key + ((iz >> j) & 1)
      out[i] = key

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
    for i in range(x.shape[0]):
      key = zorder[i]
      ix = iy = iz = 0
      for j in range(0, scale.bits) :
        iz = iz + ((key & 1) << j)
        key = key >>1
        iy = iy + ((key & 1) << j)
        key = key >>1
        ix = ix + ((key & 1) << j)
        key = key >>1
      x[i] = ix / (scale._norm[0]) + scale._min[0]
      y[i] = iy / (scale._norm[1]) + scale._min[1]
      z[i] = iz / (scale._norm[2]) + scale._min[2]
