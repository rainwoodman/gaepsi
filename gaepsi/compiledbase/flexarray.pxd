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
  cmpfunc cmpfunc
  size_t size
  size_t used
  void * ptr # used if a delegate is given as NULL

cdef inline void init(FlexArray * array, void ** delegate, int itemsize, size_t size) nogil:
  if delegate == NULL:
    array.delegate = &array.ptr
  else:
    array.delegate = delegate
  array.itemsize = itemsize
  array.size = size
  array.used = 0
  (array.delegate)[0] = malloc(itemsize * (size + 1))
  array.cmpfunc = NULL
cdef inline intptr_t append(FlexArray * array, size_t additional) nogil:
  cdef void * ptr = array.delegate[0]
  while array.used + additional > array.size:
    array.size = (array.size + 1)* 2 - 1
  (array.delegate)[0] = realloc(ptr, array.itemsize * (array.size + 1))
  cdef intptr_t index = array.used 
  array.used = array.used + additional
  return index

cdef inline void * append_ptr(FlexArray * array, size_t additional) nogil:
  cdef intptr_t index = append(array, additional)
  return &((array.delegate)[0][array.itemsize * index])

cdef inline void * get_ptr(FlexArray * array, intptr_t index) nogil:
  return &((array.delegate)[0][array.itemsize * index])

cdef inline void remove(FlexArray * array, size_t additional) nogil:
  array.used = array.used - additional

cdef inline void destroy(FlexArray * array) nogil:
  cdef void * ptr = array.delegate[0]
  free(ptr)
  array.delegate[0] = NULL

cdef inline numpy.ndarray tonumpy(FlexArray * array, dtype, object owner):
  cdef numpy.dtype mydtype = numpy.dtype(dtype)
  cdef numpy.npy_intp d = array.used * mydtype.itemsize
  cdef numpy.ndarray obj = numpy.PyArray_SimpleNewFromData(1, &d, numpy.NPY_BYTE, array.delegate[0])
  numpy.set_array_base(obj, owner)
  return obj.view(dtype=mydtype)

cdef inline void copyitem(FlexArray * array, intptr_t dst, intptr_t src) nogil:
  cdef void * ptr = array.delegate[0]
  cdef void * dstbase = <void*>ptr
  cdef void * srcbase = <void*>ptr
  if dst < 0: 
    dst = array.size + dst + 1
  if src < 0: 
    src = array.size + src + 1
  memcpy(&dstbase[dst * array.itemsize], &srcbase[src * array.itemsize], array.itemsize)

cdef inline int cmpitem(FlexArray * array, intptr_t dst, intptr_t src) nogil:
  cdef void * ptr = array.delegate[0]
  cdef char * dstbase = <char*>ptr
  cdef char * srcbase = <char*>ptr
  if dst < 0: 
    dst = array.size + dst + 1
  if src < 0: 
    src = array.size + src + 1
  return array.cmpfunc(&dstbase[dst * array.itemsize], &srcbase[src * array.itemsize])
  
cdef inline void siftdown(FlexArray * array, intptr_t startpos, intptr_t pos) nogil:
  copyitem(array, -1, pos)
  cdef intptr_t parent 
  while pos > startpos:
    parent = (pos - 1) >> 1
    if cmpitem(array, -1, parent) < 0:
      copyitem(array, pos, parent)
      pos = parent;
      continue
    else: break
  copyitem(array, pos, -1)

cdef inline void siftup(FlexArray * array, intptr_t pos) nogil:
  cdef intptr_t startpos = pos
  cdef intptr_t childpos = 2 * pos + 1
  cdef intptr_t rightpos
  copyitem(array, -1, pos)
  while childpos < array.used:
    rightpos = childpos + 1;
    if rightpos < array.used and cmpitem(array, childpos, rightpos) >=0 :
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
    i = i - 1

