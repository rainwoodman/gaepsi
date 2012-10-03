#cython: embedsignature=True
#cython: cdivision=True

cimport numpy
import numpy
cimport fillingcurve
from fillingcurve cimport fckey_t, ipos_t
from query cimport Query, node_t, TreeIter, Scratch
from libc.stdint cimport *
cimport flexarray

numpy.import_array()

cdef class NGBQueryD(Query):

  def __init__(self, tree, int ngbhint, int root=0):
    Query.__init__(self, tree, root, 'intp', ngbhint)

  def __call__(self, x, y, z, dx, dy=None, dz=None):
    if dy is None: dy = dx
    if dz is None: dz = dx
    return Query._iterover(self,
      [x, y, z, dx, dy, dz],
      ['f8'] * 6,
      [['readonly']] * 6)

  cdef void execute(self, TreeIter iter, Scratch scratch, char** data) nogil:
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

    cdef node_t node = iter.get_next_child()
    while node >= 0:
      key = self.tree.get_node_key(node)
      order = self.tree.get_node_order(node)
      flag = fillingcurve.heyinAABB(key, order, AABBkey)
      if flag == 0:
        # not intersecting
        node = iter.get_next_sibling()
      elif flag == -2:
        self._add_items(scratch, self.tree.get_node_first(node), 
           self.tree.get_node_npar(node))
        node = iter.get_next_sibling()
      elif self.tree.get_node_nchildren(node) == 0:
        for i in range(self.tree.get_node_first(node), 
           self.tree.get_node_first(node) + self.tree.get_node_npar(node), 1):
          if 0 == fillingcurve.heyinAABB(self.tree._zkey[i], 0, AABBkey): continue
          scratch.add_item(&i)
        node = iter.get_next_sibling()
      else:
        node = iter.get_next_child()

  cdef void _add_items(self, Scratch scratch, intptr_t first, intptr_t npar) nogil:
    cdef intptr_t i
    for i in range(first, first + npar, 1):
      scratch.add_item(&i)


cdef struct Element:
  intptr_t index
  double weight

cdef int elecmpfunc(Element * e1, Element * e2) nogil:
  return (e1.weight < e2.weight) - (e1.weight > e2.weight)

cdef class NGBQueryN(Query):
  def __init__(self, tree, int ngbcount, int root=0):
    Query.__init__(self, tree, root, [('indices', 'intp'), ('weights', 'f8')], ngbcount)
    Query.set_cmpfunc(self, <flexarray.cmpfunc>elecmpfunc)

  def __call__(self, x, y, z):
    return Query._iterover(self,
      [x, y, z],
      ['f8'] * 3,
      [['readonly']] * 3)

  cdef void execute(self, TreeIter iter, Scratch scratch, char** data) nogil:
    cdef double pos[3], size[3]
    cdef fckey_t AABBkey[2]
    cdef ipos_t ipos[3]
    cdef int d
    cdef fckey_t centerkey
    cdef double maxdist2, thisdist2, maxdist

    for d in range(3):
      pos[d] = (<double*>(data[d]))[0]
    
    fillingcurve.f2i(self.tree._scale, pos, ipos)
    fillingcurve.i2fc(ipos, &centerkey)

    self.tree.get_node_size(
         self.tree.get_container(pos, scratch.fa.size), 
         size)

    maxdist = size[0]
    maxdist2 = 0

    # iterate 
    while True:
      scratch.reset()
      for d in range(3):
        size[d] = maxdist
      self._getAABBkey(pos, size, AABBkey)

      self.execute_one(iter, scratch, centerkey, AABBkey)

      thisdist2 = (<Element*>scratch.get_ptr(0)).weight
      if thisdist2 <= maxdist2: break
      maxdist2 = thisdist2
      maxdist = maxdist2 ** 0.5 
      
  cdef void execute_one(self, TreeIter iter, Scratch scratch, fckey_t centerkey, fckey_t AABBkey[2]) nogil:
    cdef fckey_t key
    cdef int order 
    cdef int flag

    node = iter.get_next_child()
    while node >= 0:
      key = self.tree.get_node_key(node)
      order = self.tree.get_node_order(node)
      flag = fillingcurve.heyinAABB(key, order, AABBkey)
      if flag == 0:
        # not intersecting
        node = iter.get_next_sibling()
      elif flag == -2:
        self._add_node_weighted(scratch, centerkey, node)
        node = iter.get_next_sibling()
      elif self.tree.get_node_nchildren(node) == 0:
        self._add_node_weighted(scratch, centerkey, node)
        node = iter.get_next_sibling()
      else:
        node = iter.get_next_child()

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
  
  cdef void _add_node_weighted(self, Scratch scratch, fckey_t centerkey, node_t node) nogil:
    cdef Element e
    cdef size_t nodenpar = self.tree.get_node_npar(node)
    cdef intptr_t nodefirst = self.tree.get_node_first(node)
    for item in range(nodefirst, nodefirst + nodenpar, 1):
      e.index = item
      e.weight = fillingcurve.key2key2(self.tree._scale, centerkey, self.tree._zkey[item])
      scratch.add_item(&e)

