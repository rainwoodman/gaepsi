#cython: embedsignature=True
#cython: cdivision=True
import numpy
cimport cpython
cimport numpy
from libc.stdint cimport *
from libc.stdlib cimport malloc, realloc, free
from libc.float cimport FLT_MAX
cimport cython
import cython
from cython.parallel import prange

numpy.import_array()

cdef packed struct NodeInfo:
  int64_t key # from key and level to derive the bot and top limits
  int order
  int child_length
  intptr_t parent
  intptr_t child[8] # child[0]  may save first_par and child[1] may save npar

NodeInfoDtype = numpy.dtype([('key', 'i8'), ('order', 'i4'), ('child_length', 'i4'), ('parent', 'i8'), ('child', ('i8', 8))])

ctypedef fused floatingbuffer:
  float [:]
  double [:]

ctypedef fused floatingbuffer2:
  float [:]
  double [:]

def _(func, args):
  signature = []
  for x in args:
    if isinstance(x, (numpy.ndarray, numpy.generic)):
      if x.dtype is numpy.dtype('f4'):
        signature += ['float[:]']
      if x.dtype is numpy.dtype('f8'):
        signature += ['double[:]']
      if x.dtype is numpy.dtype('i2'):
        signature += ['short[:]']
      if x.dtype is numpy.dtype('i4'):
        signature += ['int[:]']
      if x.dtype is numpy.dtype('i8'):
        signature += ['long[:]']
    else:
      signature += [cython.typeof(x)]
  try:
    return func.__signatures__[', '.join(signature)]
  except:
    print signature, func.__signatures__
    raise

cdef int insquare(int64_t sqkey, int order, int64_t k2) nogil:
  return 0 == ((sqkey ^ k2) >> (order * 3))

cdef class Result(object):
  cdef intptr_t * _buffer
  cdef readonly size_t used
  cdef readonly size_t size
  def __cinit__(self):
    self.size = 8
    self._buffer = <intptr_t *>malloc(sizeof(intptr_t) * self.size)
  def __init__(self):
    self.used = 0
  def getarray(self):
    cdef numpy.intp_t dims[1]
    dims[0] = self.used
    arr = numpy.PyArray_SimpleNewFromData(1, dims, numpy.NPY_INTP, self._buffer)
    numpy.set_array_base(arr, self)
    return arr.view(dtype=numpy.intp)

  @cython.boundscheck(False)
  cdef void _grow(self) nogil:
    if self.size < 1024576:
      self.size *= 2
    else:
      self.size += 1024576
    self._buffer = <intptr_t *>realloc(self._buffer, sizeof(intptr_t) * self.size)

  cdef void truncate(self) nogil:
    self.used = 0

  @cython.boundscheck(False)
  cdef void append_one(self, intptr_t i) nogil:
    if self.size - self.used <= 1:
      self._grow()
    self._buffer[self.used] = i
    self.used = self.used + 1
    
  @cython.boundscheck(False)
  cdef void append(self, intptr_t start, intptr_t length) nogil:
    cdef intptr_t i
    while self.size - self.used <= length:
      self._grow()
    for i in range(start, start + length):
      self._buffer[self.used] = i
      self.used = self.used + 1

  def __dealloc__(self):
    free(self._buffer)

cdef class Scale(object):
  """Scale scales x,y,z to 0 ~ (1<<bits) - 1 """
  cdef readonly numpy.ndarray min, norm
  cdef readonly int bits
  def __cinit__(self):
    pass
  def __init__(self, x=None, y=None, z=None, min=None, norm=None, bits=1):
    if min is not None:
      self.min = numpy.asarray(min)
      self.norm = numpy.asarray(norm)
    elif x is not None:
      self.min = numpy.array([x.min(), y.min(), z.min()], dtype=x.dtype)
      self.norm = numpy.array([
         1 / (x.max() - self.min[0]),
         1 / (y.max() - self.min[1]),
         1 / (z.max() - self.min[2])], dtype=x.dtype)
    else:
      self.min = numpy.array([0., 0., 0.])
      self.norm = numpy.array([1., 1., 1.])
    self.bits = bits
    self.norm *= ((1 << bits) - 1)

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
  cdef NodeInfo * _buffer
  cdef readonly size_t size
  cdef readonly size_t used
  cdef readonly size_t thresh
  cdef readonly numpy.ndarray zorder
  cdef readonly int bits
  cdef readonly Scale scale

  def __cinit__(self):
    self.size = 1024
    self._buffer = <NodeInfo *>malloc(sizeof(NodeInfo) * self.size)
    self.used = 0

  @cython.boundscheck(False)
  def __init__(self, xyz=None, zorder=None, scale=None, thresh=1):
    """ if zorder = None, xyz = (x, y, z) will be sorted in place
        if zorder is not None, assume the input is already ordered
        scale it transforms the position to [0, 1]
    """
    self.thresh = thresh
    if zorder is None and xyz is not None:
      x, y, z = xyz
      self.zorder = numpy.empty(shape=x.shape[0], dtype='i8')
      self.bits = self.zorder.dtype.itemsize * 8 / 3
      self.scale = Scale(x, y, z, bits=self.bits)
      _zorder(x, y, z, out=self.zorder, scale=self.scale)
      ind = self.zorder.argsort()
      tmp = x[ind]
      x[:] = tmp
      tmp = y[ind]
      y[:] = tmp
      tmp = z[ind]
      z[:] = tmp
      self.zorder = self.zorder[ind]
    elif zorder is not None and xyz is None and scale is not None:
      self.zorder = zorder
      self.scale = scale
    else:
      raise ValueError("give xyz, or zorder and scale")

    self.bits = self.zorder.dtype.itemsize * 8 / 3
    self._tree_build()

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
    arr = numpy.PyArray_SimpleNewFromData(1, dims, numpy.NPY_INTP, self._buffer[ind].child)
    numpy.set_array_base(arr, self)
    rt = dict(key=self._buffer[ind].key, 
            order=self._buffer[ind].order,
           parent=self._buffer[ind].parent, 
            index=ind,
         children=arr)

    if self._buffer[ind].child_length == 0:
       rt.update(dict(first=self._buffer[ind].child[0], 
                      last=self._buffer[ind].child[0] + self._buffer[ind].child[1]))
    else:
       rt.update(dict(first=0, last=-1))
    return rt
    
  @cython.boundscheck(False)
  def query_neighbours(Tree self, x, y, z, int32_t count):
    cdef numpy.ndarray[numpy.intp_t, ndim=2] out
    cdef intptr_t i, d
    cdef int64_t [:] zorder = self.zorder 
    cdef int64_t [:] queryzorder
    cdef float center[3]
    cdef Result result
    oshape = numpy.array(x).shape

    x = numpy.atleast_1d(x)
    y = numpy.atleast_1d(y)
    z = numpy.atleast_1d(z)
    queryzorder = _zorder(x, y, z, scale=self.scale)

    out = numpy.zeros((x.shape[0], count), dtype=numpy.intp)
    for i in range(x.shape[0]):
      result = Result()
      center[0] = x[i]
      center[1] = y[i]
      center[2] = z[i]
      self.__query_neighbours_one(zorder, result, queryzorder[i], center, count, out[i, :])
    return out

  @cython.boundscheck(False)
  cdef void __query_neighbours_one(Tree self, int64_t [:] zorder, Result result, int64_t key, float center[3], int32_t count, intptr_t[:] out) except *:
    cdef intptr_t chain[32]
    cdef int last = 0
    cdef intptr_t this, child
    cdef intptr_t i, j, k
    cdef int64_t[:] keys
    # chain is the list of all nodes that contain this point
    cdef numpy.ndarray[float, ndim=1] x, y, z
    cdef float largest_distance = FLT_MAX

   
    with nogil:

      chain[last] = 0
      this = chain[last]
      last = 1
      while self._buffer[this].child_length > 0:
        for i in range(self._buffer[this].child_length):
          child = self._buffer[this].child[i]
          if insquare(self._buffer[child].key, self._buffer[child].order, key):
            chain[last] = child
            last = last + 1
            this = child
            break
    
      for i in range(last-1, -1, -1):
        result.truncate()
        self._add_all(result, chain[i])
        if result.used < count: continue
        with gil:
          keys = numpy.empty(result.used, dtype=numpy.int64)

        for k in range(result.used):
          keys[k] = zorder[result._buffer[k]]

        with gil:
          x, y, z = zorder_inverse(keys, scale=self.scale)

        for k in range(x.shape[0]):
          x[k] -= center[0]
          y[k] -= center[1]
          z[k] -= center[2]
          x[k] = x[k] * x[k] + y[k] * y[k] + z[k] * z[k]
        with gil:
          keys = x.argsort()
        if largest_distance <= x[keys[count - 1]]:
          break
        else:
          for k in range(count):
            out[k] = result._buffer[keys[k]]
          largest_distance = x[keys[count - 1]]

  @cython.boundscheck(False)
  def query_box(Tree self, x, y, z, radius):
    ix0, iy0, iz0 = self.scale(x, y, z)
    oshape = numpy.asarray(x).shape
    ir = numpy.empty(3, dtype='i4')
    ir[:] = radius * self.scale.norm
    cdef int32_t [:] ix = numpy.atleast_1d(ix0)
    cdef int32_t [:] iy = numpy.atleast_1d(iy0)
    cdef int32_t [:] iz = numpy.atleast_1d(iz0)

    cdef int32_t min[3]
    cdef int32_t max[3]
    cdef int32_t center[3]
    cdef Result result
    cdef intptr_t i, d
    cdef numpy.ndarray[cython.object, ndim=1] ret = numpy.zeros(ix.shape[0], dtype=numpy.dtype('object'))

    for i in range(ix.shape[0]):
      center[0] = ix[i]
      center[1] = iy[i]
      center[2] = iz[i]
      for d in range(3):
        min[d] = center[d] - ir[d]
        max[d] = center[d] + ir[d]

      result = Result()
      self.__query_box_one(result, min, max, 0)
      ret[i] = result.getarray()

    return ret.reshape(oshape)

  @cython.boundscheck(False)
  cdef void __query_box_one(Tree self, Result result, int32_t min[3], int32_t max[3], intptr_t root) nogil:
    cdef int64_t key
    cdef int32_t cmin[3]
    cdef intptr_t i, j, d
    cdef int32_t box
    cdef int64_t [:] zorder = self.zorder
    key = self._buffer[root].key
    box = (1 << self._buffer[root].order) - 1
    cmin[0] = 0
    cmin[1] = 0
    cmin[2] = 0
    for j in range(0, self.bits) :
      cmin[2] += ((key & 1) << j)
      key >>= 1
      cmin[1] += ((key & 1) << j)
      key >>= 1
      cmin[0] += ((key & 1) << j)
      key >>= 1

    for d in range(3):
      if cmin[d] > max[d] or cmin[d] + box < min[d]:
        return 

    verygood = 1
    for d in range(3):
      if cmin[d] < min[d] or cmin[d] + box> max[d]:
        verygood = 0
        break
    if verygood:
      self._add_all(result, root)
      return

    if self._buffer[root].child_length > 0:
      for i in range(self._buffer[root].child_length):
        self.__query_box_one(result, min, max, self._buffer[root].child[i])
    else:
      for i in range(self._buffer[root].child[0], self._buffer[root].child[0] + self._buffer[root].child[1]):
        key = zorder[i]
        cmin[0] = 0
        cmin[1] = 0
        cmin[2] = 0
        for j in range(0, self.bits) :
          cmin[2] += ((key & 1) << j)
          key >>= 1
          cmin[1] += ((key & 1) << j)
          key >>= 1
          cmin[0] += ((key & 1) << j)
          key >>= 1
        verygood = 1
        for d in range(3):
          if cmin[d] > max[d] or cmin[d] < min[d]:
            verygood = 0
            break
        if verygood: result.append_one(i)

  @cython.boundscheck(False)
  cdef void _add_all(Tree self, Result result, intptr_t ind) nogil:
    cdef intptr_t k
    if self._buffer[ind].child_length == 0:
      result.append(self._buffer[ind].child[0], self._buffer[ind].child[1])
    else:
      for k in range(self._buffer[ind].child_length):
        self._add_all(result, self._buffer[ind].child[k])

  @cython.boundscheck(False)
  def _tree_build(Tree self):
    cdef intptr_t j = 0
    cdef intptr_t i = 0
    cdef int64_t [:] zorder = self.zorder
    with nogil:
      self.used = 1;
      self._buffer[0].key = 0
      self._buffer[0].child[0] = 0
      self._buffer[0].child[1] = 0
      self._buffer[0].order = self.bits
      self._buffer[0].parent = -1
      self._buffer[0].child_length = 0
      while i < self.zorder.shape[0]:
        while not insquare(self._buffer[j].key, self._buffer[j].order, zorder[i]):
          j = self._buffer[j].parent
          #/* because we are on a morton key ordered list, no need to deccent into children */
        if self._buffer[j].child_length > 0 or (self._buffer[j].child[1] >= self.thresh and self._buffer[j].order > 0):
          if(self._buffer[j].child_length == 0):
#                  /* splitting a leaf */
            i = self._buffer[j].child[0]
#             /* otherwise appending to a new child */
          j = self._create_child(zorder, i, j) 
         # NOTE: i is rewinded!
        else:
#            /* put the particle into the leaf. */
          self._buffer[j].child[1] = self._buffer[j].child[1] + 1
        i = i + 1

  @cython.boundscheck(False)
  cdef intptr_t _create_child(self, int64_t[:] zorder, intptr_t first_par, intptr_t parent) nogil:
    # creates a child of parent from first_par, returns the new child */
    self._buffer[self.used].child[0]= first_par
    self._buffer[self.used].child[1] = 1
    self._buffer[self.used].parent = parent
    self._buffer[self.used].child_length = 0
    self._buffer[self.used].order = self._buffer[parent].order - 1
    #/* the lower bits of a sqkey is cleared off but I don't think it is necessary */
    self._buffer[self.used].key = (zorder[first_par] >> (self._buffer[self.used].order * 3)) << (self._buffer[self.used].order * 3)
    self._buffer[parent].child[self._buffer[parent].child_length] = self.used
    self._buffer[parent].child_length = self._buffer[parent].child_length + 1
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

  _(_fused_zorder, [x, scale.min])(out, x, y, z, scale.min, scale.norm, bits)
     
  return out
zorder = _zorder

@cython.boundscheck(False)
def _fused_zorder(
   int64_t[:] out,
   floatingbuffer x,
   floatingbuffer y,
   floatingbuffer z,
   floatingbuffer2 min,
   floatingbuffer2 norm,
   int bits
):
  cdef intptr_t i, j
  # do not use sizeof
  cdef int32_t ix, iy, iz
  cdef int64_t key

  for i in prange(x.shape[0], nogil=True):
      ix = <int32_t> (norm[0] * (x[i] - min[0]))
      iy = <int32_t> (norm[1] * (y[i] - min[1]))
      iz = <int32_t> (norm[2] * (z[i] - min[2]))
      key = 0
      for j in range(bits-1, -1, -1) :
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

  _(_fused_zorder_inverse, [x, scale.min])(zorder, x, y, z, scale.min, scale.norm, scale.bits)
  return x, y, z

@cython.boundscheck(False)
def _fused_zorder_inverse(
   int64_t [:] zorder,
   floatingbuffer x,
   floatingbuffer y,
   floatingbuffer z,
   floatingbuffer2 min,
   floatingbuffer2 norm,
   int bits
):
  cdef intptr_t i, j
  cdef int32_t ix, iy, iz
  cdef int64_t key
  for i in prange(x.shape[0], nogil=True):
    key = zorder[i]
    ix = iy = iz = 0
    for j in range(0, bits) :
      iz = iz + ((key & 1) << j)
      key = key >>1
      iy = iy + ((key & 1) << j)
      key = key >>1
      ix = ix + ((key & 1) << j)
      key = key >>1
    x[i] = ix / (norm[0]) + min[0]
    y[i] = iy / (norm[1]) + min[1]
    z[i] = iz / (norm[2]) + min[2]
