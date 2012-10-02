#cython: embedsignature=True
#cython: cdivision=True
import numpy
cimport numpy
cimport npyiter
cimport npyarray
cimport npyufunc
from libc.stdint cimport *
from libc.math cimport M_1_PI, cos, sin, sqrt, fabs, acos, nearbyint
from warnings import warn
cimport fillingcurve
from fillingcurve cimport fckey_t
from fillingcurve import fckeytype
from geometry cimport DieseFunktion
from geometry cimport DieseFunktionFrustum
from geometry cimport rotate_vector
from ztree cimport node_t, Tree, TreeIter

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

ctypedef float (*kernelfunc)(float x, float y) nogil

DEF SPLINEFACTOR = 1.0 # 1.0603145160926968 # magic
cdef float splinekern(float x, float y) nogil:
  cdef double h2 = x * x + y * y
  cdef double h4 = h2 * h2
  cdef double h
  cdef double h3
  if h2 < 0.25:
    return 1.909859317102744 -10.23669021 * h2 \
          -23.27182034 * h4 *(h2 - 1)
    #   1.909859 = 6 / pi
  if h2 < 1.0:
    h = sqrt(h2)
    h3 = h2 * h
# -1/9. * 7/ pi * (1-x) **4 + 100 / 7. / pi * (1-x) **3 - 1.5 / pi * (1-x) ** 2 + 6 / 17. / 5 / pi* (1-x)
    return -0.2475743559207261 * h4 \
       - 3.556986664656963 * h3 \
       + 11.67894130021956 * h2 \
       - 11.71909411592771* h \
       + 3.84471383628584
  else:
    return 0.0
# NOTE: factor functions are not used.

cdef float splinefactor(float x, float y) nogil:
  return 1.0603145160926968

DEF CUBEFACTOR = 1.0 / (4.0)
cdef float cubekern(float x, float y) nogil:
  return 1.0
cdef float cubefactor(float x, float y) nogil:
  return (x + 1) * (y + 1) / 4.0

DEF SPHEREFACTOR = 1.0 / (3.1416 / 2)
cdef float spherekern(float x, float y) nogil:
  return 1 - (x * x + y * y)
cdef float spherefactor(float x, float y) nogil:
  # fixme
  return 1.0 / (3.1416/2)

DEF DIAMONDFACTOR = 1.0 / 2
cdef float diamondkern(float x, float y) nogil:
  x = fabs(x)
  y = fabs(y)
  return 1 - 0.5 * (x + y)
cdef float diamondfactor(float x, float y) nogil:
  # fixme
  return 1.0 / 2

DEF CROSSFACTOR = 1.0 / 3
cdef float crosskern(float x, float y) nogil:
  return 1 - fabs(x * y)
cdef float crossfactor(float x, float y) nogil:
  # fixme
  return 1.0 / 3

cdef dict KERNELS = {
'spline': (<intptr_t> splinekern, SPLINEFACTOR),
'cube': (<intptr_t> cubekern, CUBEFACTOR),
'sphere': (<intptr_t> spherekern, SPHEREFACTOR),
}

cdef void _register():
  cdef intptr_t func
  for key in KERNELS:
    func, factor = KERNELS[key]
    npyufunc.register(globals(), <void*>func, NULL,
           2, 'none', key, key)
numpy.import_array()
numpy.import_ufunc()
_register()
 
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
  cdef kernelfunc kernel_func
  # normalization factor of the kernel, integration of kernel_factor * kernel = 1.0 
  cdef double kernel_factor

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
    self.kernel_factor = SPLINEFACTOR
    self.kernel_func = splinekern
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
    """ rotate the camera, moving the target"""
    if axis is None: axis = self.up
    axis = numpy.asarray(axis)
    self.lookat(target=self.pos + rotate_vector(self.target - self.pos, axis, angle),
            up=rotate_vector(self.up, axis, angle))

  def orbit(self, angle, up=None):
    """ orbit the camera, fixing the target"""
    if up is None: up = self.up
    up = numpy.asarray(up)
    self.lookat(pos=self.target + rotate_vector(self.pos - self.target, up, angle))
    
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
    DieseFunktionFrustum(self.frustum, self.matrix)
  
  cdef int mask_object_one(self, double center[3], double r[3]) nogil:
    cdef int j
    cdef double AABB[8][3]
    for j in range(8):
      AABB[j][0] = center[0] + r[0] * _AABBshifting[j][0]
      AABB[j][1] = center[1] + r[1] * _AABBshifting[j][1]
      AABB[j][2] = center[2] + r[2] * _AABBshifting[j][2]
    return DieseFunktion(self._frustum, AABB)
  
  def __call__(self, x, y, z, out=None):
    return self.transform(x, y, z, out)

  def paint(self, x, y, z, r, color, luminosity, numpy.ndarray out=None, kernel='spline'):
    if out is None:
      out = numpy.zeros(self.shape, dtype=('f8', 2))
    self.kernel_func = <kernelfunc> <intptr_t>KERNELS[kernel][0]
    self.kernel_factor = <double> KERNELS[kernel][1]
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

  def prunetree(self, Tree tree, out=None, bint return_nodes=True):
    """ scan the tree and find the nodes that are inside FOV,
        returns a mask, 0 if node is out, 2 if node is over resolved,
        and 1 if the node shall be painted as a particle. 
        if return_particles is true, return a mask for particles
        """
    cdef int nchildren
    cdef size_t start = tree.node_length
    cdef double size[3]
    cdef double pos[3]
    cdef TreeIter iter = TreeIter(tree)

    cdef numpy.ndarray mask 
    if return_nodes:
      mask = numpy.zeros(len(tree), 'int8')
    else:
      mask = numpy.zeros(len(tree.zkey), 'int8')
    cdef char * _mask = <char*> mask.data

    cdef node_t parent
    cdef int d
    # mask = 0 if node is out of view
    # mask = 1 if node is unresolved  (paint them as a particle)
    # mask = 2 if node is over resolved. (do not paint them)

    # visit first child of root
    node = iter.get_next_child()
    with nogil:
      while node >= 0:

        tree.get_node_pos(node, pos)
        tree.get_node_size(node, size)
        for d in range(3): 
          pos[d] += size[d] * 0.5
          size[d] *= 1.733 # sml = 2 * size * 1.733

        children = tree.get_node_children(node, &nchildren)
#        with gil: print 'top', top, j, tree[j].order, tree[j].key, tree[j].children
        # in FOV?
        flag = self.mask_object_one(pos, size)
        if flag == 0:
          if return_nodes:
            _mask[node] = 0
          else:
            for i in range(tree.get_node_first(node), 
               tree.get_node_first(node) + tree.get_node_npar(node), 1):
              _mask[i] = 0
          node = iter.get_next_sibling()
        elif nchildren == 0:
          if return_nodes:
            _mask[node] = 1
          else:
            for i in range(tree.get_node_first(node), 
               tree.get_node_first(node) + tree.get_node_npar(node), 1):
              _mask[i] = 1
          node = iter.get_next_sibling()
        else:
          _mask[node] = 2
          node = iter.get_next_child()

    if out is None: 
      out = mask == 1
    else:
      out[:] = mask == 1
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
    op_dtypes = ['f8', 'f8', 'f8', tmp.dtype]

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
          pos[0] = (<double*>citer.data[0])[0]
          pos[1] = (<double*>citer.data[1])[0]
          pos[2] = (<double*>citer.data[2])[0]
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
    DieseFunktionFrustum(self.frustum, self.matrix)

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
      imin[d] = <int>(xy[d] - dxy[d] + 0.5)
      imax[d] = <int>(xy[d] + dxy[d] + 0.5)
      if imax[d] > imin[d] + 1: 
        imax[d] = imax[d] - 1
      di[d] = imax[d] - imin[d] + 1
      if imin[d] < 0: imin[d] = 0
      if imin[d] >= self._shape[d]: imin[d] = self._shape[d] - 1
      if imax[d] < 0: imax[d] = 0
      if imax[d] >= self._shape[d]: imax[d] = self._shape[d] - 1

    # normfac is the area of a pixel
    cdef float normfac = 4. / (di[0] * di[1]) #( dxy[0] * dxy[1])

    # bit is the base light on a pixel. adjusted by
    # the kernel intergration factor
    # integrate(tmp1 in [-1, 1], tmp2 in [-1, 1], tmp3)
    # tmp3 is the light added to the pixel
    #
    # it is pi / 2 for the spheric kernel:
    #   tmp3 = 1 - (tmp1 ** 2 + tmp2 ** 2)
    # it is 2 for the diamond kernel:
    #   tmp3 = 1 - 0.5 * (|tmp1| + |tmp2|)
    # it is 3 for the cross kernel:
    #   tmp3 = 1 - |tmp1| * |tmp2|
    # cross kernel gives vertical and horizontal
    # lines.

    cdef float bit = brightness * normfac * self.kernel_factor
    cdef float tmp1, tmp2 # always in -1 and 1, distance to center
    cdef float tmp3  # pixel value
    cdef float tmp1fac = 2.0 / di[0]
    cdef float tmp2fac = 2.0 / di[1]
    cdef intptr_t p, q

    p = (imin[0] * self._shape[1] + imin[1]) * 2

    if di[0] == 1 and di[1] == 1:
      ccd[p] += color * brightness 
      ccd[p + 1] += brightness 
      return

    cdef int cachept
    DEF CACHESIZE = 1024
    cdef float cache[CACHESIZE]
    cdef double sum = 0.0

    cachept = 0
    tmp1 = (imin[0] - xy[0]) * tmp1fac
    ix = imin[0]

    while ix <= imax[0]:
      tmp2 = (imin[1] - xy[1]) * tmp2fac
      iy = imin[1]
      q = p
      while iy <= imax[1]:
        tmp3 = self.kernel_func(tmp1, tmp2)
        if cachept < CACHESIZE:
          cache[cachept] = tmp3
          cachept = cachept + 1
        sum += tmp3
        iy = iy + 1
        tmp2 += tmp2fac
        q += 2
        # 2 = 1(for color) + 1(for luminosity)
      tmp1 += tmp1fac
      p += self._shape[1] * 2
      ix = ix + 1

    cachept = 0
    p = (imin[0] * self._shape[1] + imin[1]) * 2
    tmp1 = (imin[0] - xy[0]) * tmp1fac
    ix = imin[0]

    while ix <= imax[0]:
      tmp2 = (imin[1] - xy[1]) * tmp2fac
      iy = imin[1]
      q = p
      while iy <= imax[1]:
        if cachept < CACHESIZE:
          tmp3 = cache[cachept]
          cachept = cachept + 1
        else:
          tmp3 = self.kernel_func(tmp1, tmp2)
        if tmp3 > 0:
          ccd[q] += color * bit * tmp3
          ccd[q + 1] += bit * tmp3
        iy = iy + 1
        tmp2 += tmp2fac
        q += 2
        # 2 = 1(for color) + 1(for luminosity)
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
    return c3inv
