#cython: cdivision=True
#cython: embedsignature=True
cimport numpy
import numpy
cimport npyiter
from libc.math cimport floor, ceil
numpy.import_array()

cdef void update_min_max(double value, double * min, double * max) nogil:
  if value < min[0]: min[0] = value
  if value > max[0]: max[0] = value

cdef class Rotation:
  cdef readonly numpy.ndarray q
  cdef readonly numpy.ndarray center
  cdef double (*_q)[3]
  cdef double * _center
  def __cinit__(self):
    self.q = numpy.empty((3, 3), dtype='f8')
    self._q = <double[3] *> self.q.data
    self.center = numpy.empty(3, dtype='f8')
    self._center = <double*>self.center.data

  def __init__(self, q=numpy.eye(3), center=(0, 0, 0)):
    """ q is the rotation on row vectors, dotted from the right 
    """
    self.q[...] = q
    self.center[...] = center

  def invert(self, X, Y=None, Z=None):
    if Y is None and Z is None:
      Z = X[..., 2]
      Y = X[..., 1]
      X = X[..., 0]

    iter = numpy.nditer([X, Y, Z],
           op_flags=[['readwrite']] * 3,
           op_dtypes=['f8'] * 3,
           flags = ['zerosize_ok', 'external_loop', 'buffered'],
           casting = 'unsafe')

    cdef npyiter.CIter citer
    cdef size_t size = npyiter.init(&citer, iter)
    cdef double *x, *y, *z

    with nogil:
      while size > 0:
        while size > 0:
          x = <double*>(citer.data[0])
          y = <double*>(citer.data[1])
          z = <double*>(citer.data[2])
          self.irotate_one(x, y, z)
          size = size - 1
          npyiter.advance(&citer)
        size = npyiter.next(&citer)
    return

  def apply(self, X, Y=None, Z=None):
    if Y is None and Z is None:
      Z = X[..., 2]
      Y = X[..., 1]
      X = X[..., 0]
     
    iter = numpy.nditer([X, Y, Z],
           op_flags=[['readwrite']] * 3,
           op_dtypes=['f8'] * 3,
           flags = ['zerosize_ok', 'external_loop', 'buffered'],
           casting = 'unsafe')

    cdef npyiter.CIter citer
    cdef size_t size = npyiter.init(&citer, iter)
    cdef double *x, *y, *z

    with nogil:
      while size > 0:
        while size > 0:
          x = <double*>(citer.data[0])
          y = <double*>(citer.data[1])
          z = <double*>(citer.data[2])
          self.rotate_one(x, y, z)
          size = size - 1
          npyiter.advance(&citer)
        size = npyiter.next(&citer)
    return
    
  cdef void rotate_one(self, double *x, double *y, double *z) nogil:
    cdef double xx, yy, zz
    cdef double * _center = self._center
    x[0] -= _center[0]
    y[0] -= _center[1]
    z[0] -= _center[2]
    xx = x[0] * self._q[0][0] + y[0] * self._q[1][0] + z[0] * self._q[2][0]
    yy = x[0] * self._q[0][1] + y[0] * self._q[1][1] + z[0] * self._q[2][1]
    zz = x[0] * self._q[0][2] + y[0] * self._q[1][2] + z[0] * self._q[2][2]
    x[0] = xx + _center[0]
    y[0] = yy + _center[1]
    z[0] = zz + _center[2]

  cdef void irotate_one(self, double *x, double *y, double *z) nogil:
    cdef double xx, yy, zz
    cdef double * _center = self._center
    x[0] -= _center[0]
    y[0] -= _center[1]
    z[0] -= _center[2]
    xx = x[0] * self._q[0][0] + y[0] * self._q[0][1] + z[0] * self._q[0][2]
    yy = x[0] * self._q[1][0] + y[0] * self._q[1][1] + z[0] * self._q[1][2]
    zz = x[0] * self._q[2][0] + y[0] * self._q[2][1] + z[0] * self._q[2][2]
    x[0] = xx + _center[0]
    y[0] = yy + _center[1]
    z[0] = zz + _center[2]

cdef class Cubenoid:
  cdef readonly Rotation rotation
  cdef readonly numpy.ndarray tries
  cdef readonly numpy.ndarray oldboxsize
  cdef readonly numpy.ndarray oldorigin
  cdef readonly numpy.ndarray newboxsize
  cdef readonly numpy.ndarray neworigin
  cdef readonly numpy.ndarray center
  cdef readonly numpy.ndarray edges
  cdef double (*_edges)[3]

  def __cinit__(self):
    self.oldboxsize = numpy.empty(3, 'f8')
    self.newboxsize = numpy.empty(3, 'f8')
    self.neworigin = numpy.empty(3, 'f8')
    self.oldorigin = numpy.empty(3, 'f8')
    self.center = numpy.empty(3, 'f8')
    self.edges = numpy.empty((3, 3), dtype='f8')
    self._edges = <double[3] *> self.edges.data

  def __init__(self, matrix, origin=0, boxsize=1, center=0, neworigin=None):
    
    if not numpy.allclose(numpy.linalg.det(matrix), 1):
      raise ValueError('matrix needs to be unitary')

    self.oldboxsize[:] = boxsize
    self.oldorigin[:] = origin

    volume = numpy.product(self.oldboxsize)
    q, r = numpy.linalg.qr(matrix)
    # q as row vectors of the old basis in the new basis.
    # q is the transformation on the right, r is the new boxsize
    self.newboxsize[:] = numpy.diag(numpy.abs(r)) * volume ** 0.33333333
    self.center[:] = center
    if neworigin is None:
      self.neworigin[:] = self.center - self.newboxsize * 0.5
    else:
      self.neworigin[:] = neworigin

    self.rotation = Rotation(q, center=self.center)
    self.edges[:] = numpy.diag(self.oldboxsize).dot(q)
    self._estimate_search_range()

  def _estimate_search_range(self):
    cdef numpy.ndarray p = numpy.empty((8, 3), dtype='f8')

    p[0] = self.neworigin
    p[7] = p[0] + self.newboxsize
    p[1] = p[0] + self.newboxsize * [0, 0, 1]
    p[2] = p[0] + self.newboxsize * [0, 1, 0]
    p[3] = p[0] + self.newboxsize * [1, 0, 0]
    p[4] = p[0] + self.newboxsize * [0, 1, 1]
    p[5] = p[0] + self.newboxsize * [1, 1, 0]
    p[6] = p[0] + self.newboxsize * [1, 0, 1]

    # we never need to try to move more than the AABB of the new bounding box
    # from the perspective of the old basis.
    
    # rotate back to the original coordinate of the AABB climax points
    #
    self.rotation.invert(p)
    p -= self.oldorigin
    AABBmin = numpy.floor((p.min(axis=0))/ self.oldboxsize)
    AABBmax = numpy.ceil((p.max(axis=0))/ self.oldboxsize)
    self.tries = numpy.array(list(self._yield_tries(AABBmin, AABBmax)), dtype='f8', copy=True)
    

  def _yield_tries(self, AABBmin, AABBmax):
    oldorigininnew = self.oldorigin.copy()
    self.rotation.apply(oldorigininnew)
    tries = numpy.empty(3)
    tries[0] = AABBmin[0]
    while tries[0] <= AABBmax[0]:
      tries[1] = AABBmin[1]
      while tries[1] <= AABBmax[1]:
        tries[2] = AABBmin[2]
        while tries[2] <= AABBmax[2]:
          rect = numpy.array([ tries, 
             (tries + [0, 0, 1]),
             (tries + [0, 1, 0]),
             (tries + [1, 0, 0]),
             (tries + [1, 0, 1]),
             (tries + [1, 1, 0]),
             (tries + [0, 1, 1]),
             (tries + [1, 1, 1])])
          rect = oldorigininnew + rect.dot(self.edges)
          rectmin = rect.min(axis=0)
          rectmax = rect.max(axis=0)
         
          if not ((rectmax < self.neworigin) | (rectmin > self.neworigin+self.newboxsize)).any():
          #  print 'a', tries, rectmin, rectmax, origin, origin+boxsize
            yield tries.copy()
          else:
            #print 'r', tries, rectmin, rectmax, origin, origin+boxsize
            #yield tries.copy()
            pass

          tries[2] += 1
        tries[1] += 1
      tries[0] += 1

  def rotate(self, X, Y=None, Z=None):
    self.rotation.apply(X, Y, Z)

  def apply(self, X, Y=None, Z=None, without_rotation=False):
    """ X Y Z and fit all points into boxsize
        rot is used to determine the range of search with estimate_search_range
    """
    cdef double _origin[3]
    cdef double _boxsize[3]

    cdef double (*_tries)[3]
    _tries = <double [3]*>self.tries.data
    cdef int ntries = len(self.tries)
    # print 'total', len(self.tries)
    for d in range(3):
      _origin[d] = self.neworigin[d]
      _boxsize[d] = self.newboxsize[d]

    cdef bint norot = without_rotation
    iter = numpy.nditer([X, Y, Z, None],
           op_flags=[['readwrite']] * 3 + [['writeonly', 'allocate']],
           op_dtypes=['f8'] * 3 + ['intp'],
           flags = ['zerosize_ok', 'external_loop', 'buffered'],
           casting = 'unsafe')

    cdef npyiter.CIter citer
    cdef size_t size = npyiter.init(&citer, iter)
    cdef double *x, *y, *z
    with nogil:
      while size > 0:
        while size > 0:
          x = <double*>(citer.data[0])
          y = <double*>(citer.data[1])
          z = <double*>(citer.data[2])
          if not norot:
            self.rotation.rotate_one(x, y, z)
          (<numpy.intp_t*>(citer.data[3]))[0] = self.solve_one(x, y, z, _tries, ntries, _origin, _boxsize)
          size = size - 1
          npyiter.advance(&citer)
        size = npyiter.next(&citer)
    return iter.operands[3]

  cdef double badness(self, double pos[3], double origin[3], double boxsize[3]) nogil:
    cdef double x, badness
    badness = 0
    for d in range(3):
      x = (pos[d] - origin[d]) / boxsize[d]
      if x < 0: badness += -x
      if x >= 1.0: badness += (x - 1.0)
    return badness

  cdef int solve_one(self, double *x, double *y, double * z, double (*tries)[3], int ntries, double origin[3], double boxsize[3]) nogil:
    cdef double pos[3]
    cdef int i
    cdef double badness_min, badness
    cdef double pos_min[3]
    cdef int i_min = -1
    pos[0] = x[0]
    pos[1] = y[0]
    pos[2] = z[0]

    for i in range(ntries):
      pos[0] = x[0]
      pos[1] = y[0]
      pos[2] = z[0]
      for d in range(3):
        pos[0] += tries[i][d] * self._edges[d][0]
        pos[1] += tries[i][d] * self._edges[d][1]
        pos[2] += tries[i][d] * self._edges[d][2]

      badness = self.badness(pos, origin, boxsize)
      if badness <= 0.0:
        x[0] = pos[0]
        y[0] = pos[1]
        z[0] = pos[2]
        return i
        
      if i_min == -1 or badness < badness_min: 
        pos_min[0] = pos[0]
        pos_min[1] = pos[1]
        pos_min[2] = pos[2]
        badness_min = badness
        i_min = i

    x[0] = pos_min[0]
    y[0] = pos_min[1]
    z[0] = pos_min[2]
    return i_min

