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
  cdef fckey_t AABBkey[2]

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
    for d in range(3):
      pos[d] = (<double*>(data[d]))[0]
      size[d] = (<double*>(data[d+3]))[0]
      self.execute_one(pos, size)

  cdef void execute_one(self, double pos[3], double size[3]) nogil:
    cdef ipos_t ipos[3]
    cdef int d
    cdef double pos1[3]
    cdef double pos2[3]

    for d in range(3):
      pos1[d] = pos[d] - size[d]
      pos2[d] = pos[d] + size[d]

    fillingcurve.f2i(self.tree._scale, pos1, ipos)
    fillingcurve.i2fc(ipos, &self.AABBkey[0])
    fillingcurve.f2i(self.tree._scale, pos2, ipos)
    fillingcurve.i2fc(ipos, &self.AABBkey[1])
    fillingcurve.fc2i(self.AABBkey[1], ipos)

    self.execute_r(0)

  cdef void execute_r(self, node_t node) nogil:
    cdef int k
    cdef intptr_t i
    cdef fckey_t key = self.tree.get_node_key(node)
    cdef int order = self.tree.get_node_order(node)
    cdef int flag = fillingcurve.heyinAABB(key, order, self.AABBkey)
    cdef int nchildren
    cdef ipos_t ipos[3], ipos1[3], ipos2[3] 
    if flag == 0: return
    children = self.tree.get_node_children(node, &nchildren)

    if flag == -2:
      self.resultset.add_items_straight(self.tree.get_node_first(node), 
           self.tree.get_node_npar(node))
    else:
      if nchildren > 0:
        for k in range(nchildren):
          self.execute_r(children[k])
      else:
        for i in range(self.tree.get_node_first(node), 
           self.tree.get_node_first(node) + self.tree.get_node_npar(node), 1):
          if 0 == fillingcurve.heyinAABB(self.tree._zkey[i], 0, self.AABBkey): continue

          self.resultset.add_item_straight(i)

cdef class NGBQueryN(Query):
  cdef readonly fckey_t centerkey
  cdef fckey_t AABBkey[2]

  def __init__(self, tree, int ngbcount):
    Query.__init__(self, tree, ngbcount)

  def __call__(self, x, y, z):
    return Query._iterover(self,
      [x, y, z],
      ['f8'] * 3,
      [['readonly']] * 3)

  cdef void execute(self, char** data) nogil:
    cdef double pos[3], size[3]

    cdef int d
    for d in range(3):
      pos[d] = (<double*>(data[d]))[0]
    
    self.tree.get_node_size(
         self.tree.get_container(pos, self.resultset.fa.size), 
         size)

    self.execute_one(pos, size)
    maxdist2 = self.resultset._e[0].weight
    maxdist = maxdist2 ** 0.5
      
    while True:
      self.resultset.reset()
      for d in range(3):
        size[d] = maxdist
      self.execute_one(pos, size)
      if self.resultset._e[0].weight <= maxdist2: break
      maxdist2 = self.resultset._e[0].weight
      maxdist = maxdist2 ** 0.5 
      
  cdef void execute_one(self, double pos[3], double size[3]) nogil:
    cdef ipos_t ipos[3]
    cdef int d
    cdef double pos1[3]
    cdef double pos2[3]
    fillingcurve.f2i(self.tree._scale, pos, ipos)
    fillingcurve.i2fc(ipos, &self.centerkey)
    for d in range(3):
      pos1[d] = pos[d] - size[d]
      pos2[d] = pos[d] + size[d]
    fillingcurve.f2i(self.tree._scale, pos1, ipos)
    fillingcurve.i2fc(ipos, &self.AABBkey[0])
    fillingcurve.f2i(self.tree._scale, pos2, ipos)
    fillingcurve.i2fc(ipos, &self.AABBkey[1])
    fillingcurve.fc2i(self.AABBkey[1], ipos)

    self.execute_r(0)

  cdef void _add_node_weighted(self, node_t node) nogil:
    cdef intptr_t item
    cdef double weight
    cdef size_t nodenpar = self.tree.get_node_npar(node)
    cdef intptr_t nodefirst = self.tree.get_node_first(node)
    for item in range(nodefirst, nodefirst + nodenpar, 1):
      weight = fillingcurve.key2key2(self.tree._scale, self.centerkey, self.tree._zkey[item])
      self.resultset.add_item_weighted(item, weight)

  cdef void execute_r(self, node_t node) nogil:
    cdef int k
    cdef intptr_t i
    cdef fckey_t key = self.tree.get_node_key(node)
    cdef int order = self.tree.get_node_order(node)
    cdef int flag = fillingcurve.heyinAABB(key, order, self.AABBkey)
    cdef int nchildren
    cdef ipos_t ipos[3], ipos1[3], ipos2[3] 
    if flag == 0: return
    children = self.tree.get_node_children(node, &nchildren)
    if flag == -2 or nchildren == 0:
      self._add_node_weighted(node)
    else:
      for k in range(nchildren):
        self.execute_r(children[k])
