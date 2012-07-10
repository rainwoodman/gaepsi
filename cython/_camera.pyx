#cython: embedsignature=True
#cython: cdivision=True
import numpy
cimport cpython
cimport numpy
cimport npyiter
cimport ztree
cimport zorder
cimport npyarray
from libc.stdint cimport *
from libc.math cimport M_1_PI, cos, sin, sqrt, fabs, acos, nearbyint
from libc.string cimport memcpy
from warnings import warn
cimport cython
from zorder cimport zorder_t, _zorder_dtype
import cython

numpy.import_array()
cdef float inf = numpy.inf
cdef float nan = numpy.nan
cdef numpy.ndarray AABBshifting = (numpy.array(
       [
         [0, 0, 0],
         [0, 0, 1],
         [0, 1, 0],
         [1, 0, 0],
         [0, 1, 1],
         [1, 0, 1],
         [1, 1, 0],
         [1, 1, 1],
       ], 'f8') - 0.5) * 2
cdef double (*_AABBshifting)[3] 
_AABBshifting = <double[3]*>AABBshifting.data

cdef extern from "math.h":
  int isnan(double d) nogil

cdef class VisTree:
  cdef readonly ztree.Tree tree
  cdef ztree.NodeInfo * _nodes
  cdef readonly numpy.ndarray node_lum
  cdef readonly numpy.ndarray node_color
  cdef float * _node_lum
  cdef float * _node_color
  cdef readonly numpy.ndarray luminosity
  cdef readonly numpy.ndarray color
  cdef float [:] l_f
  cdef double [:] l_d
  cdef float [:] c_f
  cdef double [:] c_d
  cdef int fd[2]
  cdef npyarray.CArray _luminosity
  cdef npyarray.CArray _color

  def __cinit__(self, ztree.Tree tree, numpy.ndarray color, numpy.ndarray luminosity):
    self.tree = tree
    self.node_lum = numpy.zeros(shape=tree.used, dtype='f4')
    self.node_color = numpy.zeros(shape=tree.used, dtype='f4')
    self._node_lum = <float*> self.node_lum.data
    self._node_color = <float*> self.node_color.data
    self._nodes = tree._nodes
    if luminosity is None:
      self.fd[0] = -1
      self.luminosity = numpy.array(1.0)
    else:
      self.luminosity = luminosity

    npyarray.init(&self._luminosity, self.luminosity)
    
    if color is None:
      self.color = numpy.array(1.0)
    else:
      self.color = color

    npyarray.init(&self._color, self.color)
    
    self.ensure_node_lum()

    self.ensure_node_lum_r(0)
    self.ensure_node_color_r(0)
    self.node_color[...] /= self.node_lum
      
  cdef void ensure_node_lum(self):
    iter = numpy.nditer(
          [self.luminosity, self.color, self.tree.zkey], 
      op_flags=[['readonly'], ['readonly'], ['readonly']],
     op_dtypes=['f4', 'f4', _zorder_dtype],
         flags=['buffered', 'external_loop', 'zerosize_ok'], 
       casting='unsafe')
    cdef npyiter.CIter citer
    cdef size_t size = npyiter.init(&citer, iter)
    cdef intptr_t ind = -1, i
    cdef zorder_t key
    cdef float lum
    with nogil:
      while size > 0:
        while size > 0:
          key = (<zorder_t*>citer.data[2])[0]
          if ind == -1 or \
             not zorder.boxtest(self._nodes[ind].key, self._nodes[ind].order, key):
            ind = self.tree.get_container_key(key, 0)
          lum = (<float*>citer.data[0])[0]
          self._node_lum[ind] += lum
          self._node_color[ind] += (<float*>citer.data[1])[0] * lum
          npyiter.advance(&citer)
          i = i + 1
          size = size - 1
        size = npyiter.next(&citer)

  cdef float ensure_node_lum_r(self, intptr_t ind) nogil:
    if self._node_lum[ind] != 0:
      return self._node_lum[ind]
    for k in range(self._nodes[ind].child_length):
      child = self._nodes[ind].child[k]
      self._node_lum[ind] += self.ensure_node_lum_r(child)
    return self._node_lum[ind]

  cdef float ensure_node_color_r(self, intptr_t ind) nogil:
    if self._node_color[ind] != 0:
      return self._node_color[ind]
    for k in range(self._nodes[ind].child_length):
      child = self._nodes[ind].child[k]
      self._node_color[ind] += self.ensure_node_color_r(child)
    return self._node_color[ind]

  def find_large_nodes(self, Camera camera, intptr_t root, size_t thresh):
    cdef double pos[3]
    cdef double r[3]
    cdef int d, k
    self.tree.get_node_pos(root, pos)
    self.tree.get_node_size(root, r)
    
    for d in range(3):
      pos[d] += r[d] * 0.5

    if 0 == camera.mask_object_one(pos, r):
      return []
    else:
      if self._nodes[root].child_length > 0 \
      and self._nodes[root].npar > thresh:
        rt = []
        for k in range(self._nodes[root].child_length):
          rt += self.find_large_nodes(camera, self._nodes[root].child[k], thresh)
        return rt
      else:
        return [ root ]

  cdef size_t paint_node(self, Camera camera, intptr_t index, double * ccd) nogil:
    cdef float uvt[3], whl[3]
    cdef float luminosity, color
    cdef float c3inv
    cdef int d
    cdef size_t rt = 0
    cdef double pos[3]
    cdef double r[3]
    cdef int k 
    self.tree.get_node_pos(index, pos)
    self.tree.get_node_size(index, r)

    for d in range(3):
      pos[d] += r[d] * 0.5

    if 0 == camera.mask_object_one(pos, r):
      return 0

    r[0] *= 1.733
    r[1] *= 1.733
    r[2] *= 1.733
    if self._nodes[index].child_length == 0:
      for k in range(self._nodes[index].npar):
        self.tree.get_leaf_pos(self._nodes[index].first+k, pos)
        c3inv = camera.transform_one(pos, uvt)
        camera.transform_size_one(r , c3inv, whl)
        npyarray.flat(&self._luminosity, index, &luminosity)
        npyarray.flat(&self._color, index, &color)
        camera.paint_object_one(pos, 
          uvt, whl, color, luminosity, ccd)
      return self._nodes[index].npar
    else:
      # FIXME:  when r becomes anisotropic
      if (r[0] / (camera.far - camera.near) < 0.1):
        c3inv = camera.transform_one(pos, uvt)
        camera.transform_size_one(r, c3inv, whl)
        if (whl[0] * camera._hshape[0] < 0.1 and whl[1] * camera._hshape[1] < 0.1):
          camera.paint_object_one(pos, 
            uvt, whl, self._node_color[index], self._node_lum[index], ccd)
          return 1
      for k in range(self._nodes[index].child_length):
        rt += self.paint_node(camera, self._nodes[index].child[k], ccd)
      return rt

  def paint(self, Camera camera, intptr_t root=0, numpy.ndarray out=None):
    if out is None:
      out = numpy.zeros(camera.shape, dtype=('f8', 2))
    assert out.dtype == numpy.dtype('f8')
    assert (<object>out).shape[0] == camera.shape[0]
    assert (<object>out).shape[1] == camera.shape[1]
    assert (<object>out).shape[2] == 2
    cdef size_t total = 0
    with nogil:
      total = self.paint_node(camera, root, <double*> out.data)
    print 'total nodes painted:', total
    return out


cdef class Camera:
  """ One big catch is that 
      we use C arrays whilst OpenGL uses
      Fortran arrays. Arrays still look the same on
      the paper, but the indexing is different. """
  cdef readonly numpy.ndarray target
  cdef readonly numpy.ndarray pos
  cdef readonly numpy.ndarray dir
  cdef readonly numpy.ndarray up
  cdef readonly numpy.ndarray model
  cdef readonly numpy.ndarray proj
  cdef size_t _shape[2]
  cdef float _hshape[2]
  cdef readonly float near
  cdef readonly float far
  cdef readonly float fov
  cdef readonly float aspect
  cdef float l, r, b, t
  cdef readonly numpy.ndarray matrix
  cdef readonly numpy.ndarray frustum
  cdef double * _matrix
  cdef double * _proj
  cdef double * _target
  cdef double * _pos
  cdef double (*_frustum)[4]
  cdef int _fade
  # tainted by shape.set and set_proj
  # w = scale[0] * r
  # h = scale[1] * r
  # l = scale[2] * r + scale[3] * c3inv * c3inv
  # scale[0] = cm[0,0], scale[1] = cm[1,1] 
  # only one of scale[2] or scale[3] is nonzero.
  # scale[2] = cm[2,2] * cm[3,3]
  # scale[3] = cm[3,2] * cm[2,3]
  cdef float _scale[4]

  def __cinit__(self):
    self.target = numpy.array([0, 1, 0], 'f8')
    self.pos = numpy.zeros(3)
    self.up = numpy.zeros(3)
    self.dir = numpy.zeros(3)
    self.model = numpy.eye(4)
    self.proj = numpy.eye(4)
    self.matrix = numpy.eye(4)
    self.frustum = numpy.zeros((6,4), 'f8')
    self._shape[0] = 200
    self._shape[1] = 200
    self._hshape[0] = 100
    self._hshape[1] = 100
    self._target = <double*>self.target.data
    self._matrix = <double*>self.matrix.data
    self._pos = <double*>self.pos.data
    self._proj = <double*>self.proj.data
    self._frustum = <double[4]*>self.frustum.data
    self._fade = 1

  property fade:
    def __get__(self):
      """ whether the luminosity will fade """
      return self._fade != 0
    def __set__(self, value):
      if value:
        self._fade = 1
      else:
        self._fade = 0

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

  def copy(self):
    a = Camera(self.shape[0], self.shape[1])
    a.lookat(up=self.up, target=self.target, pos=self.pos)
    a.l = self.l
    a.r = self.r
    a.b = self.b
    a.t = self.t
    a.far = self.far
    a.near = self.near
    a.fov = self.fov
    a.aspect = self.aspect
    a.fade = self.fade
    a.set_proj(self.proj)
    return a

  def rotate(self, angle, axis=None):
    if axis is None: axis = self.up
    d = self.target - self.pos
    dot = d.dot(axis) 
    cross = d.cross(axis)
    cos = numpy.cos(angle)
    sin = numpy.sin(angle)
    d = - axis * dot * (1 - cos) + d * cos + cross * sin
    self.lookat(pos=self.camera.target - d)
    
  property extent:
    def __get__(self):
      return (self.l, self.r, self.b, self.t)
      
  def persp(self, near, far, fov, aspect=1.0):
    """ fov is in radian """
    self.near = near
    self.far = far
    self.fov = fov
    self.aspect = aspect
    D = ((self.target - self.pos) ** 2).sum() ** 0.5
    self.l = - numpy.tan(fov * 0.5) * aspect * D
    self.r = - self.l
    self.b = - numpy.tan(fov * 0.5) * D
    self.t = - self.b
    persp = numpy.zeros((4,4))
    persp[0, 0] = 1.0 / numpy.tan(fov * 0.5) / aspect
    persp[1, 1] = 1.0 / numpy.tan(fov * 0.5)
    persp[2, 2] = - (1. *(far + near)) / (far - near)
    persp[2, 3] = - (2. * far * near) / (far - near)
    persp[3, 2] = -1
    persp[3, 3] = 0
    self.set_proj(persp)

  def ortho(self, near, far, extent):
    """ set up the zoom by extent=(left, right, top, bottom """
    self.near = near
    self.far = far
    l, r, b, t = extent
    self.l = l
    self.r = r
    self.b = b
    self.t = t
    self.aspect = (r - l) / (t - b)
    D = ((self.target - self.pos) ** 2).sum() ** 0.5
    self.fov = numpy.arctan2(0.5 * (t - b), D) * 2
    ortho = numpy.zeros((4,4))
    ortho[0, 0] = 2.0 / (r - l)
    ortho[1, 1] = 2.0 / (t - b)
    ortho[2, 2] = -2.0 / (far - near)
    ortho[3, 3] = 1
    ortho[0, 3] = - (1. * r + l) / (r - l)
    ortho[1, 3] = - (1. * t + b) / (t - b)
    ortho[2, 3] = - (1. * far + near) / (far - near)
    self.set_proj(ortho)

  def set_proj(self, matrix):
    self.proj[...] = matrix[...]
    self._scale[0] = fabs(self.proj[0, 0])
    self._scale[1] = fabs(self.proj[1, 1])
    self._scale[2] = fabs(self.proj[2, 2] * self.proj[3, 3])
    self._scale[3] = fabs(self.proj[2, 3] * self.proj[3, 2])
    self.matrix[...] = numpy.dot(self.proj, self.model)
    self.extract_frustum()

  def extract_frustum(self):
    """by -=sinuswutz=- from glTerrian"""

    self.frustum[0,:] = self.matrix[3,:] - self.matrix[0,:] #rechte plane berechnen
    self.frustum[1,:] = self.matrix[3,:] + self.matrix[0,:] #linke plane berechnen
    self.frustum[2,:] = self.matrix[3,:] + self.matrix[1,:] #unten plane berechnen
    self.frustum[3,:] = self.matrix[3,:] - self.matrix[1,:] #oben plane berechnen
    self.frustum[4,:] = self.matrix[3,:] - self.matrix[2,:] #ferne plane berechnen
    self.frustum[5,:] = self.matrix[3,:] + self.matrix[2,:] #nahe plane berechnen
    self.frustum[...] /= ((self.frustum[:,:-1] ** 2).sum(axis=-1)**0.5)[:, None]
  
  cdef int mask_object_one(self, double center[3], double r[3]) nogil:
    cdef int j
    cdef double AABB[8][3]
    for j in range(8):
      AABB[j][0] = center[0] + r[0] * _AABBshifting[j][0]
      AABB[j][1] = center[1] + r[1] * _AABBshifting[j][1]
      AABB[j][2] = center[2] + r[2] * _AABBshifting[j][2]
    return self.mask_object_one_AABB(AABB)

  cdef inline int mask_object_one_AABB(self, double AABB[8][3]) nogil:
    """ Diese Funktion liefert 
        0 zurück, wenn die geprüften coordinaten nicht sichtbar sind,
        1 zurück, wenn die coords teilweise sichtbar sind und 
        2 zurück, wenn alle coords sichtbar sind  
        by -=sinuswutz=- from glTerrian
    """
    cdef int cnt=0, vor=0, i, j
    #cnt : zählt, bei wie vielen ebenen alle punkte davor liegen, 
    #vor: zählt für jede ebene, wieviele punkte davor liegen
    for i in range(6):  #für alle ebenen...
      vor = 0 
      for j in range(8):   #für alle punkte...
        if AABB[j][0] * self._frustum[i][0] \
         + AABB[j][1] * self._frustum[i][1] \
         + AABB[j][2] * self._frustum[i][2] \
         + self._frustum[i][3] > 0: 
          vor = vor + 1

      # alle ecken hinter der ebene, ist nicht sichtbar!
      if vor == 0: return 0 
      # alle vor der ebene, merken und weitermachen    
      if vor == 8: cnt = cnt + 1 

    #liegt komplett im frustum
    if cnt == 6: return 2  
  
    # liegt teilweise im frustum;
    return 1  
  
  def __call__(self, x, y, z, out=None):
    return self.transform(x, y, z, out)

  def paint(self, x, y, z, r, color, luminosity, numpy.ndarray out=None):
    if out is None:
      out = numpy.zeros(self.shape, dtype=('f8', 2))
    assert out.dtype == numpy.dtype('f8')
    assert (<object>out).shape[0] == self.shape[0]
    assert (<object>out).shape[1] == self.shape[1]
    assert (<object>out).shape[2] == 2

    if color is None: color = 1
    if luminosity is None: luminosity = 1
    iter = numpy.nditer(
          [x, y, z, r, color, luminosity], 
      op_flags=[['readonly'], ['readonly'], ['readonly'], 
                ['readonly'], ['readonly'], ['readonly']], 
     op_dtypes=['f8', 'f8', 'f8', 'f8', 'f4', 'f4'],
         flags=['buffered', 'external_loop', 'zerosize_ok'], 
       casting='unsafe')
    cdef npyiter.CIter citer
    cdef size_t size = npyiter.init(&citer, iter)
    cdef double * ccd = <double*> out.data
    cdef double pos[3], R[3]
    cdef float uvt[3], whl[3], c3inv
   
    with nogil:
      while size > 0:
        while size > 0:
          pos[0] = (<double*>citer.data[0])[0]
          pos[1] = (<double*>citer.data[1])[0]
          pos[2] = (<double*>citer.data[2])[0]
          R[0] = (<double*>citer.data[3])[0]
          R[1] = (<double*>citer.data[3])[0]
          R[2] = (<double*>citer.data[3])[0]
          c3inv = self.transform_one(pos, uvt)
          self.transform_size_one(R,
              c3inv, 
              whl)
          self.paint_object_one(pos,
              uvt, whl,
              (<float*>citer.data[4])[0],
              (<float*>citer.data[5])[0],
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
          flags=['buffered', 'external_loop', 'zerosize_ok'], 
          casting='unsafe')

    cdef npyiter.CIter citer
    cdef size_t size = npyiter.init(&citer, iter)
    cdef float c3inv, R, uvt[3]
    cdef double pos[3]
    with nogil: 
      while size > 0:
        while size > 0:
          pos[0] = (<float*>citer.data[0])[0]
          pos[1] = (<float*>citer.data[1])[0]
          pos[2] = (<float*>citer.data[2])[0]
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

  def mask(self, x, y, z, r, out=None):
    """ return a mask for objects within the camera frustum.
        out is integer:
        0 invisible,
        1 partially visible,
        2 fully visible.
    """
    if out is None:
      out = numpy.empty(numpy.broadcast(x,y,z).shape, dtype='u1')

    cdef int rdim
    arrays    = [x, y, z, out]
    op_flags  = [['readonly'], ['readonly'], ['readonly'], ['writeonly']]
    op_dtypes = ['f8', 'f8', 'f8', 'u1']

    if isinstance(r, tuple):
      rdim = 3
      arrays += [r[0], r[1], r[2]]
      op_flags += [['readonly']] * 3
      op_dtypes += ['f8'] * 3
    else:
      rdim = 1
      arrays += [r]
      op_flags += [['readonly']]
      op_dtypes += ['f8']

    iter = numpy.nditer(
          arrays, op_flags=op_flags, op_dtypes=op_dtypes,
          flags=['buffered', 'external_loop', 'zerosize_ok'], 
          casting='unsafe')

    cdef npyiter.CIter citer
    cdef size_t size = npyiter.init(&citer, iter)
    cdef double pos[3]
    cdef double R[3]
    with nogil: 
      while size > 0:
        while size > 0:
          pos[0] = (<double*> citer.data[0])[0]
          pos[1] = (<double*> citer.data[1])[0]
          pos[2] = (<double*> citer.data[2])[0]
          if rdim == 1:
            R[0] = (<double*> citer.data[4])[0]
            R[1] = (<double*> citer.data[4])[0]
            R[2] = (<double*> citer.data[4])[0]
          else:
            R[0] = (<double*> citer.data[4])[0]
            R[1] = (<double*> citer.data[5])[0]
            R[2] = (<double*> citer.data[6])[0]
          (<uint8_t *> citer.data[3])[0] = self.mask_object_one(pos, R)

          npyiter.advance(&citer)
          size = size - 1
        size = npyiter.next(&citer)
    return out
 
  def lookat(self, pos=None, target=None, up=None):
    if pos is None: pos = self.pos
    if target is None: target = self.target
    if up is None: up = self.up

    pos = numpy.asarray(pos)
    self.pos[...] = pos[...]
    target = numpy.asarray(target)
    self.target[...] = target[...]
    self.up[...] = numpy.asarray(up)
    self.dir[...] = target - pos
    self.dir[...] = self.dir / (self.dir **2).sum() ** 0.5
    side = numpy.cross(self.dir, up)
    side[:] = side / (side **2).sum() ** 0.5
    self.up[...] = numpy.cross(side, self.dir)
    self.up[...] = self.up / (self.up **2).sum() ** 0.5
    
    m1 = numpy.zeros((4,4))
    m1[0, 0:3] = side
    m1[1, 0:3] = self.up
    m1[2, 0:3] = -self.dir
    m1[3, 3] = 1
    
    tran = numpy.eye(4)
    tran[0:3, 3] = -pos
    m2 = numpy.dot(m1, tran)

    self.model[...] = m2[...]
    self.matrix[...] = numpy.dot(self.proj, self.model)
    self.extract_frustum()

  cdef inline void transform_size_one(self, double r[3], float c3inv, float whl[3]) nogil:
    whl[0] = r[0] * self._scale[0] * c3inv
    whl[1] = r[1] * self._scale[1] * c3inv
    whl[2] = r[2] * c3inv * (self._scale[2] + self._scale[3] * c3inv)
    
  cdef void paint_object_one(self, double pos[3], float uvt[3], float whl[3], float color, float luminosity, double * ccd) nogil:

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

    cdef float D = 0.
    cdef float DD
    # the brightness decedes with inverse square, if self.fade is True
    if self._fade != 0:
      for d in range(3):
        DD = pos[d] - self._pos[d]
        D += DD * DD
      D = D * 3.1416
    else: D = 1.0

    # times physical size of a pixel, assuming unit size
    cdef float brightness = luminosity / D 
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
    # it is pi / 2 for the spheric kernel:
    #   tmp3 = tmp1 ** 2 + tmp2 ** 2
    # it is 2 for the diamond kernel:
    #   tmp3 = 0.5 * (|tmp1| + |tmp2|)
    # it is 3 for the cross kernel:
    #   tmp3 = |tmp1| * |tmp2|
    # cross kernel gives vertical and horizontal
    # lines.

    cdef float bit = brightness * normfac / (3.1416 / 2.0)
    cdef float tmp1, tmp2, tmp3 # always in -1 and 1, distance to center
    cdef float tmp1fac = 2.0 / di[0]
    cdef float tmp2fac = 2.0 / di[1]
    cdef intptr_t p, q

    tmp1 = (imin[0] - xy[0]) * tmp1fac
    ix = imin[0]
    p = (imin[0] * self._shape[1] + imin[1]) * 2
    while ix <= imax[0]:
      tmp2 = (imin[1] - xy[1]) * tmp2fac
      iy = imin[1]
      q = p
      while iy <= imax[1]:
        tmp3 = (tmp1 ** 2 + tmp2 **2)
        if tmp3 < 1 and tmp3 > 0: 
          ccd[q] += color * bit * (1 - tmp3)
          ccd[q + 1] += bit * (1 - tmp3)
        iy = iy + 1
        tmp2 += tmp2fac
        q += 2
      tmp1 += tmp1fac
      p += self._shape[1] * 2
      ix = ix + 1

  cdef inline float transform_one(self, double pos[3], float uvt[3]) nogil:
    cdef int k
    cdef float coord[4]
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
