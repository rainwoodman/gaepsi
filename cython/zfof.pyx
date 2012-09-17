cimport numpy
import numpy

cimport fillingcurve
from fillingcurve cimport ipos_t, fckey_t
cimport ztree
import ztree
cimport zquery
import zquery
cimport cpython
from libc.stdint cimport *
from warnings import warn
cimport npyiter
cimport fillingcurve

cdef double dist2(double fpos[3], double fpos2[3]):
  cdef double D = 0
  cdef int d
  cdef double dd
  for d in range(3):
            dd = fpos[d] - fpos2[d]
            D += dd * dd
  return D
ctypedef struct data:
  intptr_t * _head
  intptr_t * _next
  intptr_t * _tail
  intptr_t * _len

cdef class FOFCluster:
  cdef readonly numpy.ndarray head
  cdef numpy.ndarray next
  cdef numpy.ndarray tail
  cdef numpy.ndarray len

  cdef intptr_t * _head
  cdef intptr_t * _next
  cdef intptr_t * _tail
  cdef intptr_t * _len
  cdef readonly ztree.Tree tree

  def __cinit__(self, ztree.Tree tree):
    self.tree = tree

  def __call__(self, double linkl):
    """ returns labels, len
        label: one integer group id per item,
        len: the length of the groups.
        the groups are sorted decsending order of length.
    """
    head, len = self.execute(linkl)
    u, labels = numpy.unique(head, return_inverse=True)
    len = len[u]
    a = len.argsort()[::-1]
    u[a] = numpy.arange(u.size)
    labels = u[labels]
    len = len[a]
    return labels, len

  # this needs gil because _head, _len , etc have a racing condition!
  cdef void execute_one(self, data * dt, zquery.Query query, intptr_t target, double linkl2):
    cdef intptr_t k
    cdef intptr_t j, p, s
    cdef ipos_t id[3]
    cdef double fd[3]
    for k in range(query.used):
      j = query._items[k]
      if dt._head[j] == dt._head[target]: continue
      if fillingcurve.key2key(self.tree._zkey[target], self.tree.zkey[j]) >= linkl2:
        continue
      if dt._len[dt._head[target]] > dt._len[dt._head[j]]:
        p = target
        s = j
      else:
        p = j
        s = target
      dt._next[dt._tail[dt._head[p]]] = dt._head[s]
      dt._tail[dt._head[p]] = dt._tail[dt._head[s]]
      dt._len[dt._head[p]] += dt._len[dt._head[s]]
      s = dt._head[s]
      while s >= 0:
        dt._head[s] = dt._head[p]
        s = dt._next[s]
    
  def execute(self, double linkl):
    cdef size_t n = self.tree.zkey.size
    cdef double linkl2 = linkl * linkl
    cdef intptr_t target
    cdef zquery.Query query = zquery.Query(limit=128, weighted=False)
    cdef double fsize[3]
    fsize[0] = linkl
    fsize[1] = linkl
    fsize[2] = linkl
    cdef double fpos[3]
    cdef intptr_t k

    cdef numpy.ndarray head = numpy.empty(n, dtype='intp')
    cdef numpy.ndarray next = numpy.empty(n, dtype='intp')
    cdef numpy.ndarray tail = numpy.empty(n, dtype='intp')
    cdef numpy.ndarray len = numpy.empty(n, dtype='intp')
    cdef data dt
    dt._head = <intptr_t *> head.data
    dt._next = <intptr_t *> next.data
    dt._tail = <intptr_t *> tail.data
    dt._len = <intptr_t *> len.data
    for k in range(n):
      dt._head[k] = k
      dt._tail[k] = k
      dt._len[k] = 1
      dt._next[k] = -1

    for target in range(n):
      self.tree.get_par_pos(target, fpos)
      with nogil: query.execute_one(self.tree, fpos, fsize)
      self.execute_one(&dt, query, target, linkl2)

    return head, len
