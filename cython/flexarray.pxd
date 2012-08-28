#cython: cdivision=True
cimport numpy
cimport cython
from libc.stdint cimport *
from libc.stdlib cimport malloc, free, realloc
cdef struct FlexArray:
  void ** delegate
  int itemsize
  size_t size
  size_t used

cdef inline void init(FlexArray * array, void ** delegate, int itemsize, size_t size) nogil:
  array.delegate = delegate
  array.itemsize = itemsize
  array.size = size
  array.used = 0
  (array.delegate)[0] = malloc(itemsize * size)

cdef inline intptr_t append(FlexArray * array, size_t additional) nogil:
  while array.used + additional > array.size:
    array.size = array.size * 2
  (array.delegate)[0] = realloc((array.delegate)[0], array.itemsize * array.size)
  cdef intptr_t index = array.used 
  array.used = array.used + additional
  return index

cdef inline void remove(FlexArray * array, size_t additional) nogil:
  array.used = array.used - additional

cdef inline void destroy(FlexArray * array) nogil:
  free((array.delegate)[0])
  (array.delegate)[0] = NULL

