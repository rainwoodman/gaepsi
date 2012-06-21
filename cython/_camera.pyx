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
from libc.math cimport M_1_PI, cos, sin, sqrt, fabs, acos
from libc.limits cimport INT_MAX, INT_MIN
from warnings import warn
cimport cython
import cython

numpy.import_array()
cdef float inf = numpy.inf

cdef class Camera:
  cdef readonly numpy.ndarray target
  cdef readonly numpy.ndarray up
  cdef readonly numpy.ndarray pos
  cdef numpy.ndarray eyem
  cdef numpy.ndarray cameram
  cdef size_t width
  cdef size_t height
  cdef readonly numpy.ndarray matrix
  cdef double * _matrix
  cdef double * _cameram
  cdef double * _pos

  def __cinit__(self):
    self.target = numpy.array([0, 1, 0])
    self.up = numpy.array([0, 0, 1])
    self.pos = numpy.zeros(3)
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

  def __init__(self, width, height):
    self.shape = (width, height)
    self.lookat(up=[0,0,1], target=[0, 1, 0], pos=[0,0,0])

  def zoom(self, *args, **kwargs):
    raise NotImplemented('this is abstract')

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
    with nogil:
      while size > 0:
        while size > 0:
          self.paint_object_one(
              (<float*>citer.data[0])[0],
              (<float*>citer.data[1])[0],
              (<float*>citer.data[2])[0],
              (<float*>citer.data[3])[0],
              (<float*>citer.data[4])[0],
              ccd)
          npyiter.advance(&citer)
          size = size - 1
        size = npyiter.next(&citer)
    return out

  def transform_objects(self, x, y, z, r, numpy.ndarray out=None):
    """ returns
        u, v, D, w, h in out, for each element in x, y, z, r
    """
    if out is None:
      out = numpy.empty(numpy.broadcast(x,y,z,r).shape, dtype=('f4', 5))

    u, v, d, w, h = out[..., 0], out[..., 1], out[..., 2], out[..., 3], out[..., 4]
    iter = numpy.nditer(
          [x, y, z, r, u, v, d, w, h], 
      op_flags=[['readonly'], ['readonly'], ['readonly'], ['readonly'],
                ['writeonly'], ['writeonly'], ['writeonly'], 
                ['writeonly'], ['writeonly'] ], 
     op_dtypes=['f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4'],
         flags=['buffered', 'external_loop'], 
       casting='unsafe')

    cdef npyiter.CIter citer
    cdef size_t size = npyiter.init(&citer, iter)

    with nogil: 
      while size > 0:
        while size > 0:
          (<float*>citer.data[6])[0] = self.transform_object_one(
                (<float*>citer.data[0])[0],
                (<float*>citer.data[1])[0],
                (<float*>citer.data[2])[0],
                (<float*>citer.data[3])[0],
                (<float*>citer.data[4]),
                (<float*>citer.data[5]),
                (<float*>citer.data[7]),
                (<float*>citer.data[8]))
          npyiter.advance(&citer)
          size = size - 1
        size = npyiter.next(&citer)
    return out
    
  def transform(self, x, y, z, out=None):
    """ calculates the viewport positions and distance**2
        (horizontal, vertical, distance**2) of input points x,y,z
        and put it into out, if the particle is closer than the near
        cut, distance**2 will be 0, if further than the far cut it will be inf.
    """
    if out is None:
      out = numpy.empty(numpy.broadcast(x,y,z).shape, dtype=('f4', 3))
    u, v, d = out[..., 0], out[..., 1], out[..., 2]
    iter = numpy.nditer(
          [x, y, z, u, v, d], 
      op_flags=[['readonly'], ['readonly'], ['readonly'], 
                ['writeonly'], ['writeonly'], ['writeonly']], 
     op_dtypes=['f4', 'f4', 'f4', 'f4', 'f4', 'f4'],
         flags=['buffered', 'external_loop'], 
       casting='unsafe')

    cdef npyiter.CIter citer
    cdef size_t size = npyiter.init(&citer, iter)

    with nogil: 
      while size > 0:
        while size > 0:
          (<float*>citer.data[5])[0] = self.transform_one(
                (<float*>citer.data[0])[0],
                (<float*>citer.data[1])[0],
                (<float*>citer.data[2])[0],
                (<float*>citer.data[3]),
                (<float*>citer.data[4]))
          npyiter.advance(&citer)
          size = size - 1
        size = npyiter.next(&citer)
    return out
 
  def lookat(self, pos, target, up):
    pos = numpy.asarray(pos)
    self.pos[:] = pos[:]
    target = numpy.asarray(target)
    self.target[:] = target[:]
    up = numpy.asarray(up)

    dir = target - pos
    dir[:] = dir / (dir **2).sum() ** 0.5
    side = numpy.cross(dir, up)
    side[:] = side / (side **2).sum() ** 0.5
    up = numpy.cross(side, dir)
    up[:] = up / (up **2).sum() ** 0.5
    
    m1 = numpy.zeros((4,4))
    m1[0, 0:3] = side
    m1[1, 0:3] = up
    m1[2, 0:3] = -dir
    m1[3, 3] = 1
    
    tran = numpy.eye(4)
    tran[0:3, 3] = -pos
    m2 = numpy.dot(m1, tran)

    self.eyem[...] = m2[...]
    self.matrix[...] = numpy.dot(self.cameram, self.eyem)

  cdef void paint_object_one(self, float x, float y, float z, float r, float luminosity, double * ccd) nogil:
    cdef float u, v, D, w, h
    D = self.transform_object_one(x, y, z, r, &u, &v, &w, &h)
    if D <= 0 or D == inf: return
 
    # the brightness decedes with inverse square, 
    # times physical size of a pixel, assuming unit size
    brightness = luminosity / (3.14 * D) 

    cdef int ix, iy, ixmax, iymax, ixmin, iymin


    if w < 1 and h < 1:
      # unresolved
      if u < 0: return
      if v < 0: return

      if u >= self.width: return
      if v >= self.height: return
      ix = <int>u
      iy = <int>v
      ccd[ix * self.height + iy] += brightness
      return 


    if u - w < 0: ixmin = 0
    elif u - w >= self.width: ixmin = self.width
    else: ixmin = <int>(u - w)
    if v - h < 0: iymin = 0
    elif v - h >= self.height: iymin = self.height
    else: iymin = <int>(v - h)

    if u + w < 0: ixmax = 0
    elif u + w >= self.width: ixmax = self.width
    else: ixmax = <int>(u + w)
    if v + h < 0: iymax = 0
    elif v + h >= self.height: iymax = self.height
    else: iymax = <int>(v + h)
    cdef float r2
    cdef float pixarea = 1 / (w * h) # in unit circle, times 4
    cdef double sum = 0
    cdef double bit = 0
    ix = ixmin
    while ix < ixmax:
      iy = iymin
      while iy < iymax:
#        r2 = ((ix - u) * (ix - u) / (w * w)) + ((iy - v) *(iy - v) / (h * h))
        r2 = 0.75
        if r2 > 1: continue
        bit = (1 - r2) * pixarea
        sum += bit
        iy = iy + 1
      ix = ix + 1

    sum = 1 / sum
    ix = ixmin
    while ix < ixmax:
      iy = iymin
      while iy < iymax:
#        r2 = ((ix - u) * (ix - u) / (w * w)) + ((iy - v) *(iy - v) / (h * h))
        r2 = 0.75
        if r2 > 1: continue
        bit = (1 - r2) * pixarea
        ccd[ix * self.height + iy] += brightness * bit * sum
         
        iy = iy + 1
      ix = ix + 1

  cdef float transform_object_one(self, float x, float y, float z, float r, float * u, float * v, float *w, float * h) nogil:
    # w h are half sizes 
    cdef int k
    cdef float coord[4]
    cdef float Z, D
    for k in range(4):
      coord[k] = x * self._matrix[4*k+0] \
               + y * self._matrix[4*k+1] \
               + z * self._matrix[4*k+2] \
                    + self._matrix[4*k+3]
       
    u[0] = (coord[0] / coord[3] + 1.0) * 0.5 * self.width
    v[0] = (coord[1] / coord[3] + 1.0) * 0.5 * self.height
    w[0] = fabs((r * self._cameram[0] / coord[3]) * self.width)
    h[0] = fabs((r * self._cameram[5] / coord[3]) * self.height)

    Z = coord[2] / coord[3]
    if Z >= 1.0:
      return inf
    elif Z <= -1.0:
      return 0
    else:
      D = (x - self._pos[0]) ** 2
      D += (y - self._pos[1]) ** 2
      D += (z - self._pos[2]) ** 2
      return D

  cdef float transform_one(self, float x, float y, float z, float * u, float * v) nogil:
    cdef int k
    cdef float coord[4]
    cdef float Z, D
    for k in range(4):
      coord[k] = x * self._matrix[4*k+0] \
               + y * self._matrix[4*k+1] \
               + z * self._matrix[4*k+2] \
                    + self._matrix[4*k+3]
       
    u[0] = (coord[0] / coord[3] + 1.0) * 0.5 * self.width
    v[0] = (coord[1] / coord[3] + 1.0) * 0.5 * self.height
    Z = coord[2] / coord[3]
    if Z >= 1.0:
      return inf
    elif Z <= -1.0:
      return 0
    else:
      D = (x - self._pos[0]) ** 2
      D += (y - self._pos[1]) ** 2
      D += (z - self._pos[2]) ** 2
      return D

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
    self.cameram[...] = persp[...]
    self.matrix[...] = numpy.dot(self.cameram, self.eyem)
    
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
    ortho[0, 3] = - (r + l) / (r - l)
    ortho[1, 3] = - (t + b) / (t - b)
    ortho[2, 3] = - (far + near) / (far - near)
    self.cameram[...] = ortho
    self.matrix[...] = numpy.dot(self.cameram, self.eyem)
