#cython: embedsignature=True
#cython: cdivision=True
import numpy
cimport numpy
cimport cpython
from ztree cimport Tree, node_t
from libc.stdint cimport *
from libc.stdlib cimport malloc, realloc, free
cimport npyiter
from libc.math cimport sqrt
from geometry cimport LiangBarsky
cimport fillingcurve
from fillingcurve cimport fckey_t, ipos_t
cimport flexarray

numpy.import_array()

ctypedef struct Element:
  intptr_t index
  double weight

cdef int elecmpfunc(Element * e1, Element * e2) nogil:
  return (e1.weight < e2.weight) - (e1.weight > e2.weight)

cdef class ResultSet:
  cdef readonly size_t limit
  cdef Element * _e
  cdef flexarray.FlexArray fa
  property used:
    def __get__(self): return self.fa.used

  property array:
    def __get__(self): return flexarray.tonumpy(&self.fa, [('indices', 'i8'), ('weights', 'f8')], self)

  property indices:
    def __get__(self): return self.array['indices']
  property weights:
    def __get__(self): return self.array['weights']

  def __cinit__(self, int limit):
    flexarray.init(&self.fa, <void**>&self._e, sizeof(Element), limit)
    self.fa.cmpfunc = <flexarray.cmpfunc> elecmpfunc
    self.limit = limit

  def __dealloc__(self):
    flexarray.destroy(&self.fa)

  cdef void reset(self) nogil:
    self.fa.used = 0

  cdef void add_item_straight(self, intptr_t item, double weight) nogil:
    cdef Element * newitem = <Element*>flexarray.append_ptr(&self.fa, 1)
    newitem.index = item
    newitem.weight = weight

  cdef void add_items_straight(self, intptr_t first, size_t npar) nogil:
    cdef Element * newitem = <Element*>flexarray.append_ptr(&self.fa, npar)
    cdef intptr_t i
    for i in range(npar):
      newitem[i].index = first + i
      newitem[i].weight = 0.0

  cdef void add_item_weighted(self, intptr_t item, double weight) nogil:
    if self.fa.used < self.fa.size:
      self.add_item_straight(item, weight)
      if self.fa.used == self.fa.size:
        # heapify when full
        flexarray.heapify(&self.fa)
    else:
      # heap push pop
      if weight < self._e[0].weight:
        self._e[0].index = item
        self._e[0].weight = weight
        flexarray.siftup(&self.fa, 0)

cdef class Query:
  cdef readonly Tree tree
  cdef ResultSet resultset # this is a scratch.
  cdef flexarray.FlexArray indices 

  def __cinit__(self):
    flexarray.init(&self.indices, NULL, sizeof(intptr_t), 1024)

  def __init__(self, tree, limit):
    self.tree = tree
    self.resultset = ResultSet(limit)

  def __dealloc__(self):
    flexarray.destroy(&self.indices)

  cdef _iterover(self, variables, dtypes, flags):

    iter = numpy.nditer([None] + variables, 
           op_dtypes=['intp'] + dtypes,
           op_flags=[['writeonly', 'allocate']] + flags,
           flags=['zerosize_ok', 'external_loop', 'buffered'],
           casting='unsafe')
    cdef npyiter.CIter citer
    cdef size_t size = npyiter.init(&citer, iter)
    cdef intptr_t * newitems 
    with nogil:
      while size > 0:
        while size > 0:
          self.execute(citer.data + 1)
          # harvest
          newitems = <intptr_t *>flexarray.append_ptr(&self.indices, self.resultset.fa.used)

          for k in range(self.resultset.fa.used):
            newitems[k] = self.resultset._e[k].index
          (<intptr_t* >(citer.data[0]))[0] = self.resultset.fa.used
          self.resultset.reset()
          npyiter.advance(&citer)
          size = size -1
        size = npyiter.next(&citer)

    return flexarray.tonumpy(&self.indices, 'intp', self), iter.operands[0]

  cdef void execute(self, char** data) nogil:
    pass
       
cdef class NGBQueryD(Query):
  cdef fckey_t AABBkey[2]

  def __init__(self, Tree tree, int ngbhint):
    Query.__init__(self, tree, ngbhint)

  def __call__(self, x, y, z, dx, dy=None, dz=None):
    if dy is None: dy = dx
    if dz is None: dz = dx
    return self._iterover(
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

          self.resultset.add_item_straight(i, 0.0)

cdef class NGBQueryN(Query):
  cdef readonly fckey_t centerkey
  cdef fckey_t AABBkey[2]

  def __init__(self, Tree tree, int ngbcount):
    Query.__init__(self, tree, ngbcount)

  def __call__(self, x, y, z):
    return self._iterover(
      [x, y, z],
      ['f8'] * 3,
      [['readonly']] * 3)

  cdef void execute(self, char** data) nogil:
    cdef double pos[3], size[3]

    cdef int d
    for d in range(3):
      pos[d] = (<double*>(data[d]))[0]
    
    self.tree.get_node_size(
         self.tree.get_container(pos, self.resultset.limit), 
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
