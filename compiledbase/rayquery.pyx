#cython: embedsignature=True
#cython: cdivision=True

cimport numpy
import numpy
cimport fillingcurve
from fillingcurve cimport fckey_t, ipos_t
from query cimport Query, node_t, TreeIter, Scratch
from libc.stdint cimport *
from geometry cimport LiangBarsky
numpy.import_array()

cdef struct Element:
 # watchout, we store an intptr no matter what node_t is.
  intptr_t index
  double enter
  double leave

cdef class RayQueryNodes(Query):
  cdef bint full
  def __init__(self, tree, int sizehint, bint full=False):
    """ returns the complete nodes that intersects the rays.
        and enter, leave.
        if full is true, return the leave nodes, instead of the toplevel complete nodes.
    """
    Query.__init__(self, tree, [('node', 'intp'), ('enter', 'f8'), ('leave', 'f8')], sizehint)
    self.full = full

  def __call__(self, x, y, z, dir, length, root=0):
    dir = numpy.asarray(dir)
    return Query._iterover(self, root,
      [x, y, z, dir[...,0], dir[...,1], dir[...,2], length],
      ['f8'] * 7,
      [['readonly']] * 7)

  cdef void execute(self, TreeIter iter, Scratch scratch, char** data) nogil:
    cdef double center[3], dir[3], tE, tL
    cdef double pos[3], size[3]
    cdef int d
    cdef node_t node
    cdef Element e
    for d in range(3):
      center[d] = (<double *>data[d])[0]
      dir[d] = (<double *>data[d+3])[0]

    node = iter.get_next_child()
    while node >= 0:
      tE = 0
      tL = (<double *>data[6])[0]
      self.tree.get_node_pos(node, pos)
      self.tree.get_node_size(node, size)
      if LiangBarsky(pos, size, center, dir, &tE, &tL):
        nchildren = self.tree.get_node_nchildren(node)
        if ((not self.full) and nchildren == 8) or nchildren == 0:
          e.index = node
          e.enter = tE
          e.leave = tL
          scratch.add_item(&e)
          node = iter.get_next_sibling()
        else:
          node = iter.get_next_child()
      else:
        node = iter.get_next_sibling()

cdef class RayQuery(Query):
  cdef bint with_distance
  def __init__(self, tree, int sizehint, with_distance=False):
    self.with_distance
    if not with_distance:
      Query.__init__(self, tree, 'intp', sizehint)
    else:
      Query.__init__(self, tree, [('indices', 'intp'), ('enter', 'f8'), ('leave', 'f8')], sizehint)

  def __call__(self, x, y, z, dir, length, root=0):
    dir = numpy.asarray(dir)
    return Query._iterover(self, root,
      [x, y, z, dir[...,0], dir[...,1], dir[...,2], length],
      ['f8'] * 7,
      [['readonly']] * 7)

  cdef void execute(self, TreeIter iter, Scratch scratch, char** data) nogil:
    cdef double center[3], dir[3], tE, tL, tLmax
    cdef double pos[3], size[3]
    cdef int d
    cdef node_t node
    for d in range(3):
      center[d] = (<double *>data[d])[0]
      dir[d] = (<double *>data[d+3])[0]

    node = iter.get_next_child()
    while node >= 0:
      tE = 0
      tL = (<double *>data[6])[0]
      self.tree.get_node_pos(node, pos)
      self.tree.get_node_size(node, size)
      if LiangBarsky(pos, size, center, dir, &tE, &tL):
        nchildren = self.tree.get_node_nchildren(node)
        if nchildren == 0:
          self._add_items(scratch, self.tree.get_node_first(node), 
                                        self.tree.get_node_npar(node), tE, tL)
          node = iter.get_next_sibling()
        else:
          node = iter.get_next_child()
      else:
        node = iter.get_next_sibling()

  cdef void _add_items(self, Scratch scratch, intptr_t first, intptr_t npar, double tE, double tL) nogil:
    cdef Element e
    for i in range(first, first + npar, 1):
      e.index = i
      e.enter = tE
      e.leave = tL
      scratch.add_item(&i)
