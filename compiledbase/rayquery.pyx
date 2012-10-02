#cython: embedsignature=True
#cython: cdivision=True

cimport numpy
import numpy
cimport fillingcurve
from fillingcurve cimport fckey_t, ipos_t
from query cimport Query, node_t
from libc.stdint cimport *
from geometry cimport LiangBarsky
numpy.import_array()

cdef class RayQuery(Query):
  cdef readonly bint return_nodes
  def __init__(self, tree, int sizehint, return_nodes):
    """ if return_nodes is True, return the complete nodes.
        A complete node is a node has either 8 or 0 children.
        if return_nodes is False, return the particles
        in the leaf nodes that intersect with the children.
    """
    Query.__init__(self, tree, sizehint)
    self.return_nodes = return_nodes

  def __call__(self, x, y, z, dir, length):
    dir = numpy.asarray(dir)
    return Query._iterover(self,
      [x, y, z, dir[...,0], dir[...,1], dir[...,2], length],
      ['f8'] * 7,
      [['readonly']] * 7)

  cdef void execute(self, char** data) nogil:
    cdef double center[3], dir[3], tE, tL, tLmax
    cdef double pos[3], size[3]
    cdef int d
    cdef node_t node
    for d in range(3):
      center[d] = (<double *>data[d])[0]
      dir[d] = (<double *>data[d+3])[0]

    node = self.iter.get_next_child()
    while node >= 0:
      tE = 0
      tL = (<double *>data[6])[0]
      self.tree.get_node_pos(node, pos)
      self.tree.get_node_size(node, size)
      if LiangBarsky(pos, size, center, dir, &tE, &tL):
        nchildren = self.tree.get_node_nchildren(node)
        if not self.return_nodes:
          if nchildren == 0:
            self.resultset.add_items_straight(self.tree.get_node_first(node), 
                                          self.tree.get_node_npar(node))
            node = self.iter.get_next_sibling()
          else:
            node = self.iter.get_next_child()
        else:
          if nchildren == 8 or nchildren == 0:
            self.resultset.add_item_straight(node)
            node = self.iter.get_next_sibling()
          else:
            node = self.iter.get_next_child()
      else:
        node = self.iter.get_next_sibling()

