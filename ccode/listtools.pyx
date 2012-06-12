#cython: embedsignature=True
#cython: cdivision=True
import numpy
cimport cpython
from libc.stdint cimport *
cimport cython
import cython
from warnings import warn

cdef class repeat:
  """
    returns a list-like object, equivlant to itertools.repeat,
    but can be sliced and randomly accessed.
  """
  cdef object element
  cdef object times
  cdef size_t length
  cdef intptr_t offset
  def __init__(self, element, times=None):
    self.element = element
    self.times = times
    if times is None:
      self.length = INTPTR_MAX
    else:
      self.length = times

  def __getitem__(self, index):
    if isinstance(index, slice):
      if index.stop is not None:
        start, stop, step = index.indices(self.length)
        times = (stop - start + step - 1) / step
      else:
        times = None
      return repeat(self.element, times)
    else:
      return self.element

  def __repr__(self):
    return '[ %s ] * %d' % (repr(self.element), self.length)

  def __iter__(self):
    cdef intptr_t offset = 0
    while offset < self.length:
      yield self.element
      offset = offset + 1

  def __len__(self):
    return self.length

cdef class cycle:
  cdef list pool
  cdef intptr_t offset
  cdef intptr_t step
  cdef size_t pool_length
  def __init__(self, pool, offset=0, step=1):
    self.pool = pool
    self.pool_length = len(pool)
    self.offset = offset % len(pool)
    self.step = step

  def __iter__(self):
    cdef intptr_t offset = self.offset
    while True:
      yield self.pool[offset]
      offset = offset + self.step
      if offset >= self.pool_length: offset -= self.pool_length

  def __getitem__(self, index):
    if isinstance(index, slice):
      start, stop, step = index.indices(INTPTR_MAX)
      return cycle(self.pool, offset=self.offset + start * self.step, step=self.step * step)
    else:
      return self.pool[(index * self.step + self.offset) % self.pool_length]
  def __len__(self):
    return INTPTR_MAX
  def __repr__(self):
    return 'cycle(%s, %d, %d)' %(repr(self.pool), self.offset, self.step) 
cdef class zip:
  cdef size_t length
  cdef list sequences
  def __init__(self, *args):
    self.sequences = list(args)
    self.length = min([len(s) for s in self.sequences])
  def __iter__(self):
    cdef intptr_t offset = 0
    while offset < self.length:
      yield self[offset]
      offset = offset + 1

  def __getitem__(self, index):
    if isinstance(index, slice):
      return zip(*[s[index] for s in self.sequences])
    else:
      return tuple((s[index] for s in self.sequences))

  def __len__(self):
    return self.length


