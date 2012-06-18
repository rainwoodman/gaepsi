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
  cdef int width
  cdef int height
  cdef readonly numpy.ndarray matrix
  cdef double * _matrix
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

  property shape:
    def __get__(self):
      return (self.width, self.height)
    def __set__(self, value):
      self.width = value[0]
      self.height = value[1]

  def __init__(self, width, height):
    self.width= width
    self.height = height
    self.lookat(up=[0,0,1], target=[0, 1, 0], pos=[0,0,0])

  def zoom(self, *args, **kwargs):
    raise NotImplemented('this is abstract')

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
    
  def __call__(self, x, y, z, out=None):
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

    cdef npyiter.NpyIter * citer = npyiter.GetNpyIter(iter)
    cdef npyiter.IterNextFunc next = npyiter.GetIterNext(citer, NULL)
    cdef char ** data = npyiter.GetDataPtrArray(citer)
    cdef numpy.npy_intp *strides = npyiter.GetInnerStrideArray(citer)
    cdef numpy.npy_intp *size_ptr = npyiter.GetInnerLoopSizePtr(citer)
    cdef intptr_t iop, size

    with nogil: 
     while True:
      size = size_ptr[0]
      while size > 0:
        (<float*>data[5])[0] = self.transform_one(
                (<float*>data[0])[0],
                (<float*>data[1])[0],
                (<float*>data[2])[0],
                (<float*>data[3]),
                (<float*>data[4]))
        for iop in range(6):
          data[iop] += strides[iop]
        size = size - 1
      if next(citer) == 0: break
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
