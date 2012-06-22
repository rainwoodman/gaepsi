#cython: embedsignature=True
#cython: cdivision=True
# cython: profile=True
import numpy
cimport cpython
cimport numpy
cimport npyiter
cimport ztree
from libc.stdint cimport *
from libc.math cimport M_1_PI, cos, sin, sqrt, fabs, acos, nearbyint
from warnings import warn
cimport cython
import cython

numpy.import_array()
cdef float inf = numpy.inf
cdef float nan = numpy.nan

cdef extern from "math.h":
  int isnan(double d) nogil

cdef class VisTree:
  cdef ztree.Tree tree
  cdef ztree.NodeInfo * _nodes
  cdef numpy.ndarray node_lum
  cdef float * _node_lum
  cdef float Inorm
  cdef numpy.ndarray luminosity
  def __cinit__(self, ztree.Tree tree, numpy.ndarray luminosity):
    self.tree = tree
    self.node_lum = numpy.zeros(shape=tree.used, dtype='f4')
    self._node_lum = <float*> self.node_lum.data
    self._nodes = tree._buffer
#    self.node_lum[:] = nan
    self.luminosity = luminosity
    self.Inorm = self.tree.zorder._Inorm
    self.ensure_node_lum()

  @cython.boundscheck(False)
  cdef void ensure_node_lum(self):
    iter = numpy.nditer(
          [self.luminosity, self.tree.zkey], 
      op_flags=[['readonly'], ['readonly']],
     op_dtypes=['f4', 'i8'],
         flags=['buffered', 'external_loop'], 
       casting='unsafe')
    cdef npyiter.CIter citer
    cdef size_t size = npyiter.init(&citer, iter)
    cdef intptr_t ind = -1, i
    cdef int64_t key
    with nogil:
      while size > 0:
        while size > 0:
          key = (<int64_t*>citer.data[1])[0]
          if ind == -1 or \
             not ztree.insquare(self._nodes[ind].key, self._nodes[ind].order, key):
            ind = self.tree.get_container_key(key, 0)
          self._node_lum[ind] += (<float*>citer.data[0])[0]
          npyiter.advance(&citer)
          i = i + 1
          size = size - 1
        size = npyiter.next(&citer)
      self.ensure_node_lum_r(0)

  cdef float ensure_node_lum_r(self, intptr_t ind) nogil:
    if self._node_lum[ind] != 0:
      return self._node_lum[ind]
    for k in range(self._nodes[ind].child_length):
      child = self._nodes[ind].child[k]
      self._node_lum[ind] += self.ensure_node_lum_r(child)
    return self._node_lum[ind]

  cdef void paint_node(self, Camera camera, intptr_t index, double * ccd):
    cdef float pos[3]
    cdef float uvt[3], whl[3]
    cdef float r
    cdef float c3inv
    cdef int d
    self.tree.get_node_pos(index, pos)
    r = self.tree.get_node_size(index)
    c3inv = camera.transform_one(pos, uvt)
    camera.transform_size_one(r, c3inv, whl)
    for d in range(3):
      if uvt[d] - whl[d] > 1.0: return
      if uvt[d] + whl[d] < -1.0: return

    if (whl[0] * camera._hshape[0] < 0.5 and whl[1] * camera._hshape[1] < 0.5):
      camera.paint_object_one(pos, 
            uvt, whl, self._node_lum[index], ccd)
    elif self._nodes[index].child_length == 0:
      camera.paint_object_one(pos, 
            uvt, whl, self._node_lum[index], ccd)
    else:
      for k in range(self._nodes[index].child_length):
        self.paint_node(camera, self._nodes[index].child[k], ccd)

  def paint(self, Camera camera, numpy.ndarray out=None):
    if out is None:
      out = numpy.zeros(camera.shape, dtype='f8')
    assert out.dtype == numpy.dtype('f8')
    assert (<object>out).shape == camera.shape
    self.paint_node(camera, 0, <double*> out.data)
    return out


cdef class Camera:
  cdef readonly numpy.ndarray target
  cdef readonly numpy.ndarray pos
  cdef readonly numpy.ndarray up
  cdef numpy.ndarray eyem
  cdef numpy.ndarray cameram
  cdef size_t _shape[2]
  cdef float _hshape[2]
  cdef readonly numpy.ndarray matrix
  cdef double * _matrix
  cdef double * _cameram
  cdef double * _pos
  # tainted by shape.set and set_camera_matrix
  # w = scale[0] * r
  # h = scale[1] * r
  # l = scale[2] * r + scale[3] * c3inv * c3inv
  # scale[0] = cm[0,0], scale[1] = cm[1,1] 
  # only one of scale[2] or scale[3] is nonzero.
  # scale[2] = cm[2,2] * cm[3,3]
  # scale[3] = cm[3,2] * cm[2,3]
  cdef float _scale[4]

  def __cinit__(self):
    self.target = numpy.array([0, 1, 0])
    self.pos = numpy.zeros(3)
    self.up = numpy.zeros(3)
    self.eyem = numpy.eye(4)
    self.cameram = numpy.eye(4)
    self.matrix = numpy.eye(4)
    self._shape[0] = 200
    self._shape[1] = 200
    self._hshape[0] = 100
    self._hshape[1] = 100
    self._matrix = <double*>self.matrix.data
    self._pos = <double*>self.pos.data
    self._cameram = <double*>self.cameram.data

  property shape:
    def __get__(self):
      return (self._shape[0], self._shape[1])
    def __set__(self, value):
      self._shape[0] = value[0]
      self._shape[1] = value[1]
      self._hshape[0] = value[0] * 0.5
      self._hshape[1] = value[1] * 0.5

  def __init__(self, width, height):
    self.shape = (width, height)
    self.lookat(up=[0,0,1], target=[0, 1, 0], pos=[0,0,0])

  def zoom(self, *args, **kwargs):
    raise NotImplemented('this is abstract')

  def set_camera_matrix(self, matrix):
    self.cameram[...] = matrix[...]
    self._scale[0] = fabs(self.cameram[0, 0])
    self._scale[1] = fabs(self.cameram[1, 1])
    self._scale[2] = fabs(self.cameram[2, 2] * self.cameram[3, 3])
    self._scale[3] = fabs(self.cameram[2, 3] * self.cameram[3, 2])
    self.matrix[...] = numpy.dot(self.cameram, self.eyem)

  def __call__(self, x, y, z, out=None):
    return self.transform(x, y, z, out)

  def paint(self, x, y, z, r, luminosity, numpy.ndarray out=None):
    if out is None:
      out = numpy.zeros(self.shape, dtype='f8')
    assert out.dtype == numpy.dtype('f8')
    assert (<object>out).shape == self.shape

    iter = numpy.nditer(
          [x, y, z, r, luminosity], 
      op_flags=[['readonly'], ['readonly'], ['readonly'], 
                ['readonly'], ['readonly']], 
     op_dtypes=['f4', 'f4', 'f4', 'f4', 'f4'],
         flags=['buffered', 'external_loop'], 
       casting='unsafe')
    cdef npyiter.CIter citer
    cdef size_t size = npyiter.init(&citer, iter)
    cdef double * ccd = <double*> out.data
    cdef float pos[3], uvt[3], whl[3], c3inv

#    with nogil:
    while size > 0:
        while size > 0:
          pos[0] = (<float*>citer.data[0])[0],
          pos[1] = (<float*>citer.data[1])[0],
          pos[2] = (<float*>citer.data[2])[0],
          c3inv = self.transform_one(pos, uvt)
          self.transform_size_one(
              (<float*>citer.data[3])[0],
              c3inv,
              whl)
          self.paint_object_one(pos,
              uvt, whl,
              (<float*>citer.data[4])[0],
              ccd)
          npyiter.advance(&citer)
          size = size - 1
        size = npyiter.next(&citer)
    return out

  def transform(self, x, y, z, r=None, out=None):
    """ calculates the viewport positions and distance**2
        (horizontal, vertical, t) of input points x,y,z
        and put it into out, if the particle is closer than the near
        t < -1 for points too near, t > 1 for poitns too far
    """
    cdef int Nout
    if r is None:
      Nout = 3
    else:
      Nout = 6
    if out is None:
      out = numpy.empty(numpy.broadcast(x,y,z).shape, dtype=('f4', Nout))

    tmp = out.view(dtype=[('data', ('f4', Nout))]).squeeze()

    arrays    = [x, y, z, tmp]
    op_flags  = [['readonly'], ['readonly'], ['readonly'], ['writeonly']]
    op_dtypes = ['f4', 'f4', 'f4', tmp.dtype]

    if r is not None:
      arrays += [r]
      op_flags += [['readonly']]
      op_dtypes += ['f4']

    iter = numpy.nditer(
          arrays, op_flags=op_flags, op_dtypes=op_dtypes,
          flags=['buffered', 'external_loop'], 
          casting='unsafe')

    cdef npyiter.CIter citer
    cdef size_t size = npyiter.init(&citer, iter)
    cdef float c3inv, R, uvt[3], pos[3]
#    with nogil: 
    while size > 0:
        while size > 0:
          pos[0] = (<float*>citer.data[0])[0],
          pos[1] = (<float*>citer.data[1])[0],
          pos[2] = (<float*>citer.data[2])[0],
          c3inv = self.transform_one(pos,
                (<float*>citer.data[3]))
          if Nout == 6:
            R = (<float*>citer.data[4])[0] 
            R *= c3inv
            (<float*>citer.data[3])[3] = R * self._scale[0]
            (<float*>citer.data[3])[4] = R * self._scale[1]
            (<float*>citer.data[3])[5] = R * (self._scale[2] + self._scale[3] * c3inv)
 
          npyiter.advance(&citer)
          size = size - 1
        size = npyiter.next(&citer)
    return out
 
  def lookat(self, pos, target, up):
    pos = numpy.asarray(pos)
    self.pos[:] = pos[:]
    target = numpy.asarray(target)
    self.target[:] = target[:]
    self.up[:] = numpy.asarray(up)

    dir = target - pos
    dir[:] = dir / (dir **2).sum() ** 0.5
    side = numpy.cross(dir, up)
    side[:] = side / (side **2).sum() ** 0.5
    self.up[:] = numpy.cross(side, dir)
    self.up[:] = self.up / (self.up **2).sum() ** 0.5
    
    m1 = numpy.zeros((4,4))
    m1[0, 0:3] = side
    m1[1, 0:3] = self.up
    m1[2, 0:3] = -dir
    m1[3, 3] = 1
    
    tran = numpy.eye(4)
    tran[0:3, 3] = -pos
    m2 = numpy.dot(m1, tran)

    self.eyem[...] = m2[...]
    self.matrix[...] = numpy.dot(self.cameram, self.eyem)

  cdef inline void transform_size_one(self, float r, float c3inv, float whl[3]) nogil:
    r *= c3inv
    whl[0] = r * self._scale[0]
    whl[1] = r * self._scale[1]
    whl[2] = r * (self._scale[2] + self._scale[3] * c3inv)
    
  cdef void paint_object_one(self, float pos[3], float uvt[3], float whl[3], float luminosity, double * ccd) nogil:

    if whl[0] > 2 or whl[1] > 2:
      # over resolved
      return
    cdef int d

    for d in range(3):
      if uvt[d] - whl[d] > 1.0: return
      if uvt[d] + whl[d] < -1.0: return

    cdef float zfac 
    cdef float xy[2]
    cdef float dxy[2]

    for d in range(2):
      xy[d] = (uvt[d] + 1.0) * self._hshape[d]
      dxy[d] = whl[d] * self._hshape[d]

    if uvt[2] + whl[2] > 1:
      if uvt[2] - whl[2] < -1:
        zfac = 1 / whl[2]  # 2 / (whl[2] + whl[2]) is inside
      else:
        zfac = (1 - uvt[2] + whl[2]) / (whl[2] + whl[2])
    else:
      if uvt[2] - whl[2] < -1:
        zfac = (uvt[2] + whl[2] - (-1 )) / (whl[2]+whl[2])
      else:
        zfac = 1

    cdef float D = 0
    cdef float DD = 0
    for d in range(3):
      DD = pos[d] - self._pos[d]
      D += DD * DD

    # the brightness decedes with inverse square, 
    # times physical size of a pixel, assuming unit size
    # FIXME: do this!
    cdef float brightness = luminosity / (3.14 * D) 
    #cdef float brightness = luminosity
    # then we reduce the brightness because some portion of the particle is
    # too far or too near.

    brightness *= zfac
    cdef int ix, iy, imax[2], imin[2], di[2]

    for d in range(2):
      imin[d] = <int>(xy[d] - dxy[d])
      imax[d] = <int>(xy[d] + dxy[d])
      di[d] = imax[d] - imin[d] + 1
      if imin[d] < 0: imin[d] = 0
      if imin[d] >= self._shape[d]: imin[d] = self._shape[d] - 1
      if imax[d] < 0: imax[d] = 0
      if imax[d] >= self._shape[d]: imax[d] = self._shape[d] - 1

    # normfac is the average luminosity of a pixel
    cdef float normfac = 1. / ( dxy[0] * dxy[1])
    # bit is the base light on a pixel. adjusted by
    # the kernel intergration factor
    # 4 - integrate(tmp1 in [-1, 1], tmp2 in [-1, 1], tmp3)
    # it is 2 for the diamond kernel:
    #   tmp3 = 0.5 * (|tmp1| + |tmp2|)
    # it is 3 for the cross kernel:
    #   tmp3 = |tmp1| * |tmp2|
    # cross kernel gives vertical and horizontal
    # lines.

    cdef float bit = brightness * normfac / 2
    cdef float tmp1, tmp2, tmp3 # always in -1 and 1, distance to center
    cdef float tmp1fac = 2.0 / di[0]
    cdef float tmp2fac = 2.0 / di[1]
    cdef intptr_t p, q

    tmp1 = (imin[0] - xy[0]) * tmp1fac
    ix = imin[0]
    p = imin[0] * self._shape[1] + imin[1]
    while ix <= imax[0]:
      tmp2 = (imin[1] - xy[1]) * tmp2fac
      iy = imin[1]
      q = p
      while iy <= imax[1]:
        # diamond kernel
        tmp3 = 0.5 * (fabs(tmp1) + fabs(tmp2))
        if tmp3 < 1 and tmp3 > 0: 
          ccd[q] += bit * (1 - tmp3)
        iy = iy + 1
        tmp2 += tmp2fac
        q += 1
      tmp1 += tmp1fac
      p += self._shape[1]
      ix = ix + 1

  cdef inline float transform_one(self, float pos[3], float uvt[3]) nogil:
    cdef int k
    cdef float coord[4]
    cdef float D
    for k in range(4):
      coord[k] = pos[0] * self._matrix[4*k+0] \
               + pos[1] * self._matrix[4*k+1] \
               + pos[2] * self._matrix[4*k+2] \
                    + self._matrix[4*k+3]
       
    cdef float c3inv = 1.0 / coord[3]
    uvt[0] = coord[0] * c3inv
    uvt[1] = coord[1] * c3inv
    uvt[2] = coord[2] * c3inv
    return c3inv;

cdef class PCamera(Camera):
  cdef float near
  cdef float far
  cdef float fov
  cdef float aspect

  def __init__(self, width, height):
    super(PCamera, self).__init__(width, height)

  def zoom(self, near, far, fov, aspect=1.0):
    """ fov is in radian """
    self.near = near
    self.far = far
    self.fov = fov
    self.aspect = aspect
    persp = numpy.zeros((4,4))
    persp[0, 0] = 1.0 / numpy.tan(fov) / aspect
    persp[1, 1] = 1.0 / numpy.tan(fov)
    persp[2, 2] = - (1. *(far + near)) / (far - near)
    persp[2, 3] = - (2. * far * near) / (far - near)
    persp[3, 2] = -1
    persp[3, 3] = 0
    self.set_camera_matrix(persp)

cdef class OCamera(Camera):
  cdef readonly float l
  cdef readonly float r
  cdef readonly float b
  cdef readonly float t
  cdef readonly float near
  cdef readonly float far
  def __init__(self, width, height):
    super(OCamera, self).__init__(width, height)
  def zoom(self, near, far, extent):
    """ set up the zoom by extent=(left, right, top, bottom """
    l, r, b, t = extent
    ortho = numpy.zeros((4,4))
    ortho[0, 0] = 2.0 / (r - l)
    ortho[1, 1] = 2.0 / (t - b)
    ortho[2, 2] = -2.0 / (far - near)
    ortho[3, 3] = 1
    ortho[0, 3] = - (1. * r + l) / (r - l)
    ortho[1, 3] = - (1. * t + b) / (t - b)
    ortho[2, 3] = - (1. * far + near) / (far - near)
    self.set_camera_matrix(ortho)
