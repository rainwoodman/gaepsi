#cython: cdivision=True
cimport numpy
cimport cython
from libc.stdint cimport *
from libc.stdlib cimport malloc, free, realloc
from libc.string cimport memcpy

ctypedef int (*cmpfunc) (void * data1, void * data) nogil

cdef struct FlexArray:
  void ** delegate
  int itemsize
  void * scratch
  cmpfunc cmpfunc
  size_t size
  size_t used

cdef inline void init(FlexArray * array, void ** delegate, int itemsize, size_t size) nogil:
  array.delegate = delegate
  array.itemsize = itemsize
  array.size = size
  array.used = 0
  (array.delegate)[0] = malloc(itemsize * size)
  array.scratch = malloc(itemsize * 2)

cdef inline intptr_t append(FlexArray * array, size_t additional) nogil:
  cdef void * ptr = array.delegate[0]
  while array.used + additional > array.size:
    array.size = array.size * 2
  (array.delegate)[0] = realloc(ptr, array.itemsize * array.size)
  cdef intptr_t index = array.used 
  array.used = array.used + additional
  return index

cdef inline void remove(FlexArray * array, size_t additional) nogil:
  array.used = array.used - additional

cdef inline void destroy(FlexArray * array) nogil:
  cdef void * ptr = array.delegate[0]
  free(ptr)
  array.delegate[0] = NULL
  free(array.scratch)

cdef inline void copyitem(FlexArray * array, intptr_t dst, intptr_t src) nogil:
  cdef void * ptr = array.delegate[0]
  cdef void * dstbase = <void*>ptr
  cdef void * srcbase = <void*>ptr
  if dst < 0: 
    dstbase = array.scratch
    dst = - dst + 1
  if src < 0: 
    srcbase = array.scratch
    src = - src + 1
  memcpy(&dstbase[dst * array.itemsize], &srcbase[src * array.itemsize], array.itemsize)

cdef inline int cmpitem(FlexArray * array, intptr_t dst, intptr_t src) nogil:
  cdef void * ptr = array.delegate[0]
  cdef void * dstbase = <void*>ptr
  cdef void * srcbase = <void*>ptr
  if dst < 0: 
    dstbase = array.scratch
    dst = - dst + 1
  if src < 0: 
    srcbase = array.scratch
    src = - src + 1
  return array.cmpfunc(&dstbase[dst * array.itemsize], &srcbase[src * array.itemsize])
  
cdef inline void siftdown(FlexArray * array, intptr_t startpos, intptr_t pos) nogil:
  copyitem(array, -1, pos)
  cdef intptr_t parent 
  while pos > startpos:
    parent = (pos - 1) >> 1
    if cmpitem(array, -1, parent):
      copyitem(array, pos, parent)
      pos = parent;
      continue
    break
  copyitem(array, pos, -1)

cdef inline void siftup(FlexArray * array, intptr_t pos) nogil:
  cdef intptr_t startpos = pos
  cdef intptr_t childpos = 2 * pos + 1
  cdef intptr_t rightpos
  copyitem(array, -1, pos)
  while childpos < array.used:
    rightpos = childpos + 1;
    if rightpos < array.used and not cmpitem(array, childpos, rightpos):
      childpos = rightpos;
    copyitem(array, pos, childpos)
    pos = childpos;
    childpos = 2 * pos + 1;
  copyitem(array, pos, -1)
  siftdown(array, startpos, pos)

cdef inline void heapify(FlexArray * array) nogil:
  cdef intptr_t i = array.used / 2 - 1
  while i >=0 :
    siftup(array, i)

