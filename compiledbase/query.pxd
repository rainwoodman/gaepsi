from ztree cimport Tree, node_t
from libc.stdint cimport *
import numpy
cimport flexarray
cimport npyiter

ctypedef struct Element:
  intptr_t index
  double weight

cdef class ResultSet:
  cdef Element * _e
  cdef flexarray.FlexArray fa

  cdef inline void reset(self) nogil:
    self.fa.used = 0

  cdef inline void add_item_straight(self, intptr_t item) nogil:
    cdef Element * newitem = <Element*>flexarray.append_ptr(&self.fa, 1)
    newitem.index = item
    newitem.weight = 0.0

  cdef inline void add_items_straight(self, intptr_t first, size_t npar) nogil:
    cdef Element * newitem = <Element*>flexarray.append_ptr(&self.fa, npar)
    cdef intptr_t i
    for i in range(npar):
      newitem[i].index = first + i
      newitem[i].weight = 0.0

  cdef inline void add_item_weighted(self, intptr_t item, double weight) nogil:
    """ will never grow """
    cdef Element * newitem 
    if self.fa.used < self.fa.size:
      newitem = <Element*>flexarray.append_ptr(&self.fa, 1)
      newitem.index = item
      newitem.weight = weight
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
  # to be overidden
  cdef void execute(self, char** data) nogil

  cdef inline tuple _iterover(self, variables, dtypes, flags):

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

