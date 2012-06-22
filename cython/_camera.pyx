#cython: embedsignature=True
#cython: cdivision=True
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
  cdef float [:] _lum_f
  cdef double [:] _lum_d
  cdef int d_or_f
  cdef numpy.ndarray luminosity
  def __cinit__(self, ztree.Tree tree, numpy.ndarray luminosity):
    self.tree = tree
    self.node_lum = numpy.empty(shape=tree.used, dtype='f4')
    self._node_lum = <float*> self.node_lum.data
    self._nodes = tree._buffer
    self.node_lum[:] = nan
    self.Inorm = self.tree.zorder._Inorm
    if luminosity.dtype == numpy.dtype('f4'):
      self._lum_f = luminosity
      self.d_or_f = 0
    elif luminosity.dtype == numpy.dtype('f8'):
      self._lum_d = luminosity
      self.d_or_f = 1
    else:
      raise TypeError('luminosity type error')

  @cython.boundscheck(False)
  cdef float ensure_node_lum(self, intptr_t index) nogil:
    cdef double s = 0
    cdef int k

    if isnan(self._node_lum[index]):
      if self._nodes[index].child_length > 0:
        for k in range(self._nodes[index].child_length):
          s += self.ensure_node_lum(self._nodes[index].child[k])
      else:
        if self.d_or_f == 0:
          for k in range(self._nodes[index].npar):
            s += self._lum_f[self._nodes[index].first + k]
        else:
          for k in range(self._nodes[index].npar):
            s += self._lum_d[self._nodes[index].first + k]
      self._node_lum[index] = s
      return s
    else:
      return self._node_lum[index]

  cdef void paint_node(self, Camera camera, intptr_t index, double * ccd) nogil:
    cdef float pos[3]
    cdef float uvt[3], whl[3]
    cdef float r
    self.tree.get_node_pos(index, pos)
    r = self.tree.get_node_size(index) * 0.5 
    camera.transform_object_one(pos[0], pos[1], pos[2], r, uvt, whl)

    if (whl[0] < 1 and whl[1] < 1):
      camera.paint_object_one(pos[0], pos[1], pos[2], 
            uvt, whl, self.ensure_node_lum(index), ccd)
    elif self._nodes[index].child_length == 0:
      camera.paint_object_one(pos[0], pos[1], pos[2], 
            uvt, whl, self.ensure_node_lum(index), ccd)
    else:
      for k in range(self._nodes[index].child_length):
        self.paint_node(camera, self._nodes[index].child[k], ccd)

  def paint(self, Camera camera, numpy.ndarray out=None):
    if out is None:
      out = numpy.zeros(camera.shape, dtype='f8')
    assert out.dtype == numpy.dtype('f8')
    assert out.shape[0] == camera.width
    assert out.shape[1] == camera.height
    self.paint_node(camera, 0, <double*> out.data)
    return out


cdef class Camera:
  cdef readonly numpy.ndarray target
  cdef readonly numpy.ndarray pos
  cdef readonly numpy.ndarray up
  cdef numpy.ndarray eyem
  cdef numpy.ndarray cameram
  cdef size_t width
  cdef size_t height
  cdef readonly numpy.ndarray matrix
  cdef double * _matrix
  cdef double * _cameram
  cdef double * _pos
  # tainted by shape.set and set_camera_matrix
  cdef float _uscale  
  cdef float _vscale
  cdef float _tscale

  def __cinit__(self):
    self.target = numpy.array([0, 1, 0])
    self.pos = numpy.zeros(3)
    self.up = numpy.zeros(3)
    self.eyem = numpy.eye(4)
    self.cameram = numpy.eye(4)
    self.matrix = numpy.eye(4)
    self.width = 200
    self.height = 200
    self._matrix = <double*>self.matrix.data
    self._pos = <double*>self.pos.data
    self._cameram = <double*>self.cameram.data

  property shape:
    def __get__(self):
      return (self.width, self.height)
    def __set__(self, value):
      self.width = value[0]
      self.height = value[1]
      self._uscale = fabs(self.cameram[0, 0] * self.width)
      self._vscale = fabs(self.cameram[1, 1] * self.height)

  def __init__(self, width, height):
    self.shape = (width, height)
    self.lookat(up=[0,0,1], target=[0, 1, 0], pos=[0,0,0])

  def zoom(self, *args, **kwargs):
    raise NotImplemented('this is abstract')

  def set_camera_matrix(self, matrix):
    self.cameram[...] = matrix[...]
    self._uscale = fabs(self.cameram[0, 0] * self.width)
    self._vscale = fabs(self.cameram[1, 1] * self.height)
    self._tscale = fabs(self.cameram[2, 2])
    self.matrix[...] = numpy.dot(self.cameram, self.eyem)

  def __call__(self, x, y, z, out=None):
    return self.transform(x, y, z, out)

  def paint(self, x, y, z, r, luminosity, numpy.ndarray out=None):
    if out is None:
      out = numpy.zeros(self.shape, dtype='f8')
    assert out.dtype == numpy.dtype('f8')
    assert out.shape[0] == self.width
    assert out.shape[1] == self.height

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
    cdef float uvt[3], whl[3]

    with nogil:
      while size > 0:
        while size > 0:
          self.transform_object_one(
              (<float*>citer.data[0])[0],
              (<float*>citer.data[1])[0],
              (<float*>citer.data[2])[0],
              (<float*>citer.data[3])[0],
              uvt, whl)
          self.paint_object_one(
              (<float*>citer.data[0])[0],
              (<float*>citer.data[1])[0],
              (<float*>citer.data[2])[0],
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
    cdef float c3inv, R, uvt[3]
    with nogil: 
      while size > 0:
        while size > 0:
          c3inv = self.transform_one(
                (<float*>citer.data[0])[0],
                (<float*>citer.data[1])[0],
                (<float*>citer.data[2])[0],
                (<float*>citer.data[3]))
          if Nout == 6:
            R = (<float*>citer.data[4])[0] 
            R *= c3inv
            (<float*>citer.data[3])[3] = R * self._uscale
            (<float*>citer.data[3])[4] = R * self._vscale
            (<float*>citer.data[3])[5] = R * self._tscale
 
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

  cdef void transform_object_one(self, float x, float y, float z, float r, float uvt[3], float whl[3]) nogil:
    cdef float c3inv = self.transform_one(x, y, z, uvt)
    r *= c3inv
    whl[0] = r * self._uscale
    whl[1] = r * self._vscale
    whl[2] = r * self._tscale
  
  cdef void paint_object_one(self, float x, float y, float z, float uvt[3], float whl[3], float luminosity, double * ccd) nogil:
    if whl[0] > self.width or whl[1] > self.height:
      # over resolved
      return

    if uvt[3] + whl[2] < -1 or uvt[3] - whl[2] > 1: return

    cdef float zfac 

    if uvt[3] + whl[2] > 1:
      if uvt[3] - whl[2] < -1:
        zfac = 1 / whl[2]  # 2 / (whl[2] + whl[2]) is inside
      else:
        zfac = (1 - uvt[3] - whl[2]) / (whl[2] + whl[2])
    else:
      if uvt[3] - whl[2] < -1:
        zfac = (uvt[3] + whl[2] - (-1 )) / (whl[2]+whl[2])
      else:
        zfac = 1

    zfac = 1
    cdef double D = 0
    cdef double d = 0

    d = x - self._pos[0]
    D += d * d
    d = y - self._pos[1]
    D += d * d
    d = z - self._pos[2]
    D += d * d

    # the brightness decedes with inverse square, 
    # times physical size of a pixel, assuming unit size
    # FIXME: do this!
    cdef float brightness = luminosity / (3.14 * D) 
    #cdef float brightness = luminosity
    # then we reduce the brightness because some portion of the particle is
    # too far or too near.

    brightness *= zfac
    cdef int ix, iy, ixmax, iymax, ixmin, iymin

    if whl[0] < 1 and whl[1] < 1:
      # unresolved
      if uvt[0] < 0: return
      if uvt[1] < 0: return

      if uvt[0] >= self.width: return
      if uvt[1] >= self.height: return
      ix = <int>uvt[0]
      iy = <int>uvt[1]
      ccd[ix * self.height + iy] += brightness
      return 

    if uvt[0] - whl[0] < 0: ixmin = 0
    elif uvt[0] - whl[0] >= self.width: ixmin = self.width
    else: ixmin = <int>(uvt[0] - whl[0])
    if uvt[1] - whl[1] < 0: iymin = 0
    elif uvt[1] - whl[1] >= self.height: iymin = self.height
    else: iymin = <int>(uvt[1] - whl[1])

    if uvt[0] + whl[0] < 0: ixmax = 0
    elif uvt[0] + whl[0] >= self.width: ixmax = self.width
    else: ixmax = <int>(uvt[0] + whl[0])
    if uvt[1] + whl[1] < 0: iymax = 0
    elif uvt[1] + whl[1] >= self.height: iymax = self.height
    else: iymax = <int>(uvt[1] + whl[1])
    cdef float r2
    cdef double normfac = 0
    cdef double bit = 0

    # the parenthesis are important
    normfac = 4. / (nearbyint(whl[0]*2) * nearbyint(whl[1]*2))
    ix = ixmin
    while ix < ixmax:
      iy = iymin
      while iy < iymax:
        r2 = 0.75
        if r2 > 1: continue
        bit = (1 - r2) * normfac
        ccd[ix * self.height + iy] += brightness * bit
         
        iy = iy + 1
      ix = ix + 1

  cdef float transform_one(self, float x, float y, float z, float uvt[3]) nogil:
    cdef int k
    cdef float coord[4]
    cdef float D
    for k in range(4):
      coord[k] = x * self._matrix[4*k+0] \
               + y * self._matrix[4*k+1] \
               + z * self._matrix[4*k+2] \
                    + self._matrix[4*k+3]
       
    cdef float c3inv = 1.0 / coord[3]
    uvt[0] = (coord[0] * c3inv + 1.0) * 0.5 * self.width
    uvt[1] = (coord[1] * c3inv + 1.0) * 0.5 * self.height
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
