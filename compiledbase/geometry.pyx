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
  """ the perioidc boundary.
      
      a boundary can be rotated.
      then applied to positions, so that they are rotated
      together, or not, and shifted to any desired boxsize.
  """
  cdef readonly numpy.ndarray origin
  cdef readonly numpy.ndarray edges
  cdef readonly numpy.ndarray q
  cdef double (*_q)[3]
  cdef double * _origin
  cdef double (*_edges)[3]
  cdef double _edgenorm[3]
  cdef int iAABB[3][2]

  def __cinit__(self):
    self.origin = numpy.empty(3, dtype='f8')
    self._origin = <double*> self.origin.data
    self.edges = numpy.empty((3, 3), dtype='f8')
    self._edges = <double[3] *> self.edges.data
    self.q = numpy.empty((3, 3), dtype='f8')
    self._q = <double[3] *> self.q.data

  def copy(self):
    return PeriodicBoundary(self.origin, 
             self.edges[0], self.edges[1], self.edges[2])

  def __init__(self, origin, edge1=None, edge2=None, edge3=None):
    """ setup the boundary. if edge2 and edge3 .. are None,
        assume edge1 is a boxsize and the edges are Axis Aligned.
        use the boxsize as the edge vectors 
        edge vectors are the translation vectors of the boundary
    """
    self.origin[:] = origin
    if edge2 is None and edge3 is None:
      self.edges[...] = numpy.diag(edge1)
    else:
      self.edges[0] = edge1
      self.edges[1] = edge2
      self.edges[2] = edge3

    for d in range(3):
      self._edgenorm[d] = (self.edges[d] ** 2).sum() ** 0.5

    # actually the denormed edges. the initial transformation
    # where the edges are AABB.
    self.q[...] = numpy.linalg.qr(self.edges)[0]

  def cubenoid(self, matrix):
    """ cubenoid transformation, 
        returns the rotation and desired new boxsize,
        q, boxsize can be then used in apply or shift """
    if not numpy.allclose(numpy.linalg.det(matrix), 1):
      raise ValueError('matrix needs to be unitary')

    q, r = numpy.linalg.qr(matrix)
    # q as row vectors of the old basis in the new basis.
    # q is the transformation on the right, r is the new boxsize
    boxsize = numpy.diag(numpy.abs(r)) \
        * (self._edgenorm[0] \
        * self._edgenorm[1] \
        * self._edgenorm[2] ) ** 0.33333333
    return q, boxsize


  def rotate(self, q):
    """ rotate the boundary by matrix q """
    self.q[...] = self.q.dot(q)
    self.edges[...] = self.edges.dot(q)

  def apply(self, X, Y, Z, boxsize, bint apply_rotation=False):
    """ apply the transformation of the boundry to positions
        X Y Z and fit all points into boxsize
        if apply_rotation==False, assume the points X, Y, Z
        are already rotated.
    """
    cdef int iAABB[3][2]
    cdef double _boxsize[3]

    self.update_iAABB(boxsize, iAABB, _boxsize)

    iter = numpy.nditer([X, Y, Z, None],
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
          if apply_rotation:
            self.rotate_one(x, y, z)
          (<double*>(citer.data[3]))[0] = self.solve_one(x, y, z, iAABB, _boxsize)
          size = size - 1
          npyiter.advance(&citer)
        size = npyiter.next(&citer)
    return iter.operands[3]

  cdef void update_iAABB(self, boxsize, int iAABB[3][2], double _boxsize[3]):
    """ update the integer AABB search limit of the shifts """
    cdef numpy.ndarray bbinold = numpy.empty((3,3), dtype='f8')
    cdef double * _bbinold = <double*> bbinold.data

    # in the original coord, the new bounding box
    # as row vectors, q.T if the old edges are aligned to the old box.
    # FIXME: if the old edges are not aligned to the old box,
    # q.T shall not be used, but a qr of edges of some kind.
    # note that self.q.T gives the cummulated transformation back to
    # the original AABB system.
    bbinold[...] = numpy.diag(boxsize).dot(self.q.T)
    for d in range(3):
      _boxsize[d] = boxsize[d]

    cdef double AABB[3][2]
    
    # we never need to try to move more than the AABB of the new bounding box
    # from the perspective of the old basis.
    constructAABB(&_bbinold[0], &_bbinold[3], &_bbinold[6], AABB)
    for d in range(3):
      iAABB[d][0] = <int>(floor(AABB[d][0] / self._edgenorm[d]))
      iAABB[d][1] = <int>(ceil(AABB[d][1] / self._edgenorm[d]))
      #print iAABB[d][0], iAABB[d][1], AABB[d][0], self._edgenorm[d]

  cdef double badness(self, double pos[3], double boxsize[3]) nogil:
    cdef double x, badness
    badness = 0
    for d in range(3):
      x = (pos[d] - self._origin[d]) / boxsize[d]
      if x < -1e-5: badness += -x 
      if x >= 1.0 + 1e-5: badness += (x - 1.0)
    return badness

  cdef void rotate_one(self, double *x, double *y, double *z) nogil:
    cdef double xx, yy, zz
    xx = x[0] * self._q[0][0] + y[0] * self._q[1][0] + z[0] * self._q[2][0]
    yy = x[0] * self._q[0][1] + y[0] * self._q[1][1] + z[0] * self._q[2][1]
    zz = x[0] * self._q[0][2] + y[0] * self._q[1][2] + z[0] * self._q[2][2]
    x[0] = xx
    y[0] = yy
    z[0] = zz

  cdef double solve_one(self, double *x, double *y, double * z, int iAABB[3][2], double boxsize[3]) nogil:
    cdef double pos[3]
    cdef int tries[3]
    cdef double badness_min, badness
    cdef double pos_min[3]
    pos[0] = x[0]
    pos[1] = y[0]
    pos[2] = z[0]

    badness_min = self.badness(pos, boxsize)

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

          badness = self.badness(pos, boxsize)
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
