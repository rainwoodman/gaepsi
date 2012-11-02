#cython: embedsignature=True
#cython: cdivision=True
import numpy
cimport numpy
from ztree cimport Tree, node_t, TreeIter
from libc.stdint cimport *
from libc.string cimport memcpy
from libc.stdlib cimport free, malloc
cimport npyiter
from libc.math cimport sqrt
cimport flexarray

numpy.import_array()

cdef class Scratch:
  def __cinit__(self, dtype, int size):
    self.dtype = numpy.dtype(dtype)
    flexarray.init(&self.fa, NULL, self.dtype.itemsize, size)

  def __dealloc__(self):
    flexarray.destroy(&self.fa)

  property A:
    def __get__(self): return flexarray.tonumpy(&self.fa, 
       self.dtype, self)

  cdef void add_item(self, void * data) nogil:
    cdef void * newitem
    cdef int itemsize = self.dtype.itemsize
    if self.fa.cmpfunc == NULL:
      newitem = flexarray.append_ptr(&self.fa, 1)
      memcpy(&newitem[0], data, itemsize)
    else:
      if self.fa.used < self.fa.size:
        newitem = flexarray.append_ptr(&self.fa, 1)
        memcpy(&newitem[0], data, itemsize)
        if self.fa.used == self.fa.size:
          # heapify when full
          flexarray.heapify(&self.fa)
      else:
        # heap push pop
        if self.fa.cmpfunc(data, self.get_ptr(0)) > 0:
          memcpy(self.get_ptr(0), data, itemsize)
          flexarray.siftup(&self.fa, 0)
        #  with gil:
        #    print 'itemadded', self.A['weights']

cdef class _freeobj:
  cdef void * ptr
  def __dealloc__(self):
    free(self.ptr)

cdef class Query:
  """ base class for queries,
      provide an iter for iterating over the tree and a scratch for scratch
  """
  def __init__(self, tree, dtype, sizehint):
    self.tree = tree
    self.dtype = numpy.dtype(dtype)
    self.sizehint = sizehint
    self.cmpfunc = NULL

  cdef void execute(self, TreeIter iter, Scratch scratch, char** data) nogil:
    """ query.iter and query.scratch are reset before execute is called """
    pass

  cdef tuple _iterover(self, root, variables, dtypes, flags):
    cdef flexarray.FlexArray results
    cdef int itemsize = self.dtype.itemsize
    cdef Scratch scratch = Scratch(self.dtype, self.sizehint)
    cdef TreeIter treeiter = TreeIter(self.tree)

    flexarray.init(&results, NULL, itemsize, 4)

    # important 
    scratch.set_cmpfunc(self.cmpfunc)

    iter = numpy.nditer([None, root] + variables, 
           op_dtypes=['intp', 'intp'] + dtypes,
           op_flags=[['writeonly', 'allocate'], ['readonly']] + flags,
           flags=['zerosize_ok', 'external_loop', 'buffered'],
           casting='unsafe')
    cdef npyiter.CIter citer
    cdef size_t size = npyiter.init(&citer, iter)
    with nogil:
      while size > 0:
        while size > 0:
          treeiter.reset((<intptr_t* >(citer.data[1]))[0])
          self.execute(treeiter, scratch, citer.data + 2)
          # harvest
          newitems = <void *>flexarray.append_ptr(&results, scratch.fa.used)
          memcpy(newitems, scratch.fa.ptr, 
              scratch.fa.used * itemsize)
          (<intptr_t* >(citer.data[0]))[0] = scratch.fa.used
          scratch.reset()
          npyiter.advance(&citer)
          size = size - 1
        size = npyiter.next(&citer)
    owner = _freeobj()
    owner.ptr = results.ptr
    return flexarray.tonumpy(&results, self.dtype, owner), iter.operands[0]

