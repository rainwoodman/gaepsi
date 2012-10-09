#cython: cdivision=True
cimport numpy
import numpy
cimport npyiter
from libc.math cimport floor, ceil
numpy.import_array()

cdef void update_min_max(double value, double * min, double * max) nogil:
  if value < min[0]: min[0] = value
  if value > max[0]: max[0] = value

cdef void constructAABB(double e1[3], double e2[3], double e3[3], double AABB[3][2]) nogil:
  "construct the AABB of a box given by E a list of edge row vectors"
  for d in range(3):
    AABB[d][0] = 0.0
    AABB[d][1] = 0.0
    update_min_max(e1[d], &AABB[d][0], &AABB[d][1])
    update_min_max(e2[d], &AABB[d][0], &AABB[d][1])
    update_min_max(e3[d], &AABB[d][0], &AABB[d][1])
    update_min_max(e1[d] + e2[d], &AABB[d][0], &AABB[d][1])
    update_min_max(e1[d] + e3[d], &AABB[d][0], &AABB[d][1])
    update_min_max(e2[d] + e3[d], &AABB[d][0], &AABB[d][1])
    update_min_max(e1[d] + e2[d] + e3[d], &AABB[d][0], &AABB[d][1])

cdef class PeriodicBoundary:
  cdef readonly numpy.ndarray boxsize
  cdef readonly numpy.ndarray origin
  cdef readonly numpy.ndarray edges
  cdef readonly numpy.ndarray X
  cdef readonly numpy.ndarray Y
  cdef readonly numpy.ndarray Z
  cdef double * _boxsize
  cdef double * _origin
  cdef double (*_edges)[3]
  cdef double _edgenorm[3]
  def __cinit__(self):
    self.boxsize = numpy.empty(3, dtype='f8')
    self._boxsize = <double*> self.boxsize.data
    self.origin = numpy.empty(3, dtype='f8')
    self._origin = <double*> self.origin.data
    self.edges = numpy.empty((3, 3), dtype='f8')
    self._edges = <double[3] *> self.edges.data

  def copy(self):
    return PeriodicBoundary(self.origin, self.boxsize,
             self.edges[0], self.edges[1], self.edges[2])

  def attach(self, X, Y, Z):
    cdef PeriodicBoundary rt = self.copy()
    rt.X = numpy.asarray(X)
    rt.Y = numpy.asarray(Y)
    rt.Z = numpy.asarray(Z)
    return rt

  def __init__(self, origin, boxsize, edge1=None, edge2=None, edge3=None):
    """ setup the boundary. if edge1 and edge2 .. are None,
        use the boxsize as the edge vectors 
        edge vectors are the translation vectors of the boundary
    """
    self.boxsize[:] = boxsize
    self.origin[:] = origin
    if edge1 is None:
      self.edges[...] = numpy.diag(self.boxsize)
    else:
      self.edges[0] = edge1
      self.edges[1] = edge2
      self.edges[2] = edge3

    for d in range(3):
      self._edgenorm[d] = (self.edges[d] ** 2).sum() ** 0.5

  def transform(self, matrix):
    """ cubenoid transformation """
    if not numpy.allclose(numpy.linalg.det(matrix), 1):
      raise ValueError('matrix needs to be unitary')

    q, r = numpy.linalg.qr(matrix)
    # q as row vectors of the old basis in the new basis.
    # q is the transformation on the right, r is the new boxsize
    self.edges[...] = self.edges.dot(q)
    self.boxsize[...] = numpy.diag(numpy.abs(r)) \
        * (self._edgenorm[0] \
        * self._edgenorm[1] \
        * self._edgenorm[2] ) ** 0.33333333

    if self.X is None: return

    cdef numpy.ndarray trans = numpy.array(q, dtype='f8', copy=True)
    cdef double * _trans = <double*>trans.data
    cdef numpy.ndarray bbinold = numpy.empty((3,3), dtype='f8')
    cdef double * _bbinold = <double*> bbinold.data

    # in the original coord, the new bounding box
    # as row vectors, q.T if the old edges are aligned to the old box.
    # FIXME: if the old edges are not aligned to the old box,
    # q.T shall not be used, but a qr of edges of some kind.
    bbinold[...] = numpy.diag(self.boxsize).dot(numpy.linalg.qr(self.edges)[0].T)
    cdef double AABB[3][2]
    cdef int iAABB[3][2]
    
    # we never need to try to move more than the AABB of the new bounding box
    # from the perspective of the old basis.
    constructAABB(&_bbinold[0], &_bbinold[3], &_bbinold[6], AABB)
    for d in range(3):
      iAABB[d][0] = <int>(floor(AABB[d][0] / self._edgenorm[d]))
      iAABB[d][1] = <int>(ceil(AABB[d][1] / self._edgenorm[d]))
      #print iAABB[d][0], iAABB[d][1], AABB[d][0], self._edgenorm[d]

    iter = numpy.nditer([self.X, self.Y, self.Z, None],
           op_flags=[['readwrite']] * 3 + [['writeonly', 'allocate']],
           op_dtypes=['f8'] * 3 + ['f8'],
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
          self.rotate_one(x, y, z, _trans)
          (<double*>(citer.data[3]))[0] = self.solve_one(x, y, z, iAABB)
          size = size - 1
          npyiter.advance(&citer)
        size = npyiter.next(&citer)

    return iter.operands[3]
    
    
  cdef double badness(self, double pos[3]) nogil:
    cdef double x, badness
    badness = 0
    for d in range(3):
      x = (pos[d] - self._origin[d]) / self._boxsize[d]
      if x < -1e-5: badness += -x 
      if x >= 1.0 + 1e-5: badness += (x - 1.0)
    return badness

  cdef void rotate_one(self, double *x, double *y, double *z, double trans[]) nogil:
    cdef double xx, yy, zz
    xx = x[0] * trans[0] + y[0] * trans[3] + z[0] * trans[6]
    yy = x[0] * trans[1] + y[0] * trans[4] + z[0] * trans[7]
    zz = x[0] * trans[2] + y[0] * trans[5] + z[0] * trans[8]
    x[0] = xx
    y[0] = yy
    z[0] = zz

  cdef double solve_one(self, double *x, double *y, double * z, int iAABB[3][2]) nogil:
    cdef double pos[3]
    cdef int tries[3]
    cdef double badness_min, badness
    cdef double pos_min[3]
    pos[0] = x[0]
    pos[1] = y[0]
    pos[2] = z[0]

    badness_min = self.badness(pos)

    if badness_min <= 0.0: return badness_min

    tries[0] = iAABB[0][0]
    while tries[0] <= iAABB[0][1]:
      tries[1] = iAABB[1][0]
      while tries[1] <= iAABB[1][1]:
        tries[2] = iAABB[2][0]
        while tries[2] <= iAABB[2][1]:
          pos[0] = x[0]
          pos[1] = y[0]
          pos[2] = z[0]
          for d in range(3):
            pos[0] += tries[d] * self._edges[d][0]
            pos[1] += tries[d] * self._edges[d][1]
            pos[2] += tries[d] * self._edges[d][2]
          badness = self.badness(pos)
          if badness <= 0.0:
            x[0] = pos[0]
            y[0] = pos[1]
            z[0] = pos[2]
            return badness
            
          if badness < badness_min: 
            pos_min[0] = pos[0]
            pos_min[1] = pos[1]
            pos_min[2] = pos[2]
            badness_min = badness
          tries[2] += 1
        tries[1] += 1
      tries[0] += 1

    x[0] = pos_min[0]
    y[0] = pos_min[1]
    z[0] = pos_min[2]
    return badness_min
