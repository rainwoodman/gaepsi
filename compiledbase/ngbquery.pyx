#cython: embedsignature=True
#cython: cdivision=True

cimport numpy
import numpy
cimport fillingcurve
from fillingcurve cimport fckey_t, ipos_t
from query cimport Query, node_t
from libc.stdint cimport *
numpy.import_array()

cdef class NGBQueryD(Query):

  def __init__(self, tree, int ngbhint):
    Query.__init__(self, tree, ngbhint)

  def __call__(self, x, y, z, dx, dy=None, dz=None):
    if dy is None: dy = dx
    if dz is None: dz = dx
    return Query._iterover(self,
      [x, y, z, dx, dy, dz],
      ['f8'] * 6,
      [['readonly']] * 6)

  cdef void execute(self, char** data) nogil:
    cdef double pos[3], size[3]
    cdef int d
    cdef fckey_t AABBkey[2]
    for d in range(3):
      pos[d] = (<double*>(data[d]))[0]
      size[d] = (<double*>(data[d+3]))[0]

    cdef ipos_t ipos[3]
    cdef double pos1[3]
    cdef double pos2[3]

    for d in range(3):
      pos1[d] = pos[d] - size[d]
      pos2[d] = pos[d] + size[d]

    fillingcurve.f2i(self.tree._scale, pos1, ipos)
    fillingcurve.i2fc(ipos, &AABBkey[0])
    fillingcurve.f2i(self.tree._scale, pos2, ipos)
    fillingcurve.i2fc(ipos, &AABBkey[1])
    fillingcurve.fc2i(AABBkey[1], ipos)

    cdef fckey_t key
    cdef int order 
    cdef int flag

    node = self.iter.get_next_child()
    while node >= 0:
      key = self.tree.get_node_key(node)
      order = self.tree.get_node_order(node)
      flag = fillingcurve.heyinAABB(key, order, AABBkey)
      if flag == 0:
        # not intersecting
        node = self.iter.get_next_sibling()
      elif flag == -2:
        self.resultset.add_items_straight(self.tree.get_node_first(node), 
           self.tree.get_node_npar(node))
        node = self.iter.get_next_sibling()
      elif self.tree.get_node_nchildren(node) == 0:
        for i in range(self.tree.get_node_first(node), 
           self.tree.get_node_first(node) + self.tree.get_node_npar(node), 1):
          if 0 == fillingcurve.heyinAABB(self.tree._zkey[i], 0, AABBkey): continue
          self.resultset.add_item_straight(i)
        node = self.iter.get_next_sibling()
      else:
        node = self.iter.get_next_child()

cdef class NGBQueryN(Query):
  def __init__(self, tree, int ngbcount):
    Query.__init__(self, tree, ngbcount)

  def __call__(self, x, y, z):
    return Query._iterover(self,
      [x, y, z],
      ['f8'] * 3,
      [['readonly']] * 3)

  cdef void execute(self, char** data) nogil:
    cdef double pos[3], size[3]
    cdef fckey_t AABBkey[2]
    cdef ipos_t ipos[3]
    cdef int d
    cdef fckey_t centerkey

    for d in range(3):
      pos[d] = (<double*>(data[d]))[0]
    
    fillingcurve.f2i(self.tree._scale, pos, ipos)
    fillingcurve.i2fc(ipos, &centerkey)

    self.tree.get_node_size(
         self.tree.get_container(pos, self.resultset.fa.size), 
         size)

    self._getAABBkey(pos, size, AABBkey)
    self.execute_one(centerkey, AABBkey)
    maxdist2 = self.resultset._e[0].weight
    maxdist = maxdist2 ** 0.5
      
    while True:
      self.resultset.reset()
      for d in range(3):
        size[d] = maxdist
      self._getAABBkey(pos, size, AABBkey)

      self.execute_one(centerkey, AABBkey)
      if self.resultset._e[0].weight <= maxdist2: break
      maxdist2 = self.resultset._e[0].weight
      maxdist = maxdist2 ** 0.5 
      
  cdef void execute_one(self, fckey_t centerkey, fckey_t AABBkey[2]) nogil:
    cdef fckey_t key
    cdef int order 
    cdef int flag

    node = self.iter.get_next_child()
    while node >= 0:
      key = self.tree.get_node_key(node)
      order = self.tree.get_node_order(node)
      flag = fillingcurve.heyinAABB(key, order, AABBkey)
      if flag == 0:
        # not intersecting
        node = self.iter.get_next_sibling()
      elif flag == -2:
        self._add_node_weighted(centerkey, node)
        node = self.iter.get_next_sibling()
      elif self.tree.get_node_nchildren(node) == 0:
        self._add_node_weighted(centerkey, node)
        node = self.iter.get_next_sibling()
      else:
        node = self.iter.get_next_child()

  cdef void _getAABBkey(self, double pos[3], double size[3], fckey_t AABBkey[2]) nogil:
    cdef double pos1[3]
    cdef double pos2[3]
    cdef ipos_t ipos[3]
    cdef int d
    for d in range(3):
      pos1[d] = pos[d] - size[d]
      pos2[d] = pos[d] + size[d]

    fillingcurve.f2i(self.tree._scale, pos1, ipos)
    fillingcurve.i2fc(ipos, &AABBkey[0])
    fillingcurve.f2i(self.tree._scale, pos2, ipos)
    fillingcurve.i2fc(ipos, &AABBkey[1])
  
  cdef void _add_node_weighted(self, fckey_t centerkey, node_t node) nogil:
    cdef intptr_t item
    cdef double weight
    cdef size_t nodenpar = self.tree.get_node_npar(node)
    cdef intptr_t nodefirst = self.tree.get_node_first(node)
    for item in range(nodefirst, nodefirst + nodenpar, 1):
      weight = fillingcurve.key2key2(self.tree._scale, centerkey, self.tree._zkey[item])
      self.resultset.add_item_weighted(item, weight)

