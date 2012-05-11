"""
sharedmem facilities SHM parallization.
empty and wrap allocates numpy arrays on the SHM
Pool is a slave-pool that can be either based on Threads or Processes.

Notice that Pool.map and Pool.star map do not return ordered results.

"""

import multiprocessing as mp
import numpy
import os
import threading
import Queue as queue
import ctypes
import traceback
import copy_reg

from numpy import ctypeslib
from multiprocessing.sharedctypes import RawArray
from itertools import cycle, izip, repeat

def cpu_count():
  num = os.getenv("OMP_NUM_THREADS")
  try:
    return int(num)
  except:
    return mp.cpu_count()

class Pool:
  """
    with Pool() as p
      def work(a, b, c):
        pass
      p.starmap(work, zip(A, B, C))

    To use a Thread pool, pass use_threads=True
  """
  def __enter__(self):
    return self
  def __exit__(self, type, value, traceback):
    pass

  def __init__(self, np=None, use_threads=False):
    if np is None: np = cpu_count()
    self.np = np

    if use_threads:
      self.QueueFactory = queue.Queue
      self.JoinableQueueFactory = queue.Queue
      def func(*args, **kwargs):
        slave = threading.Thread(*args, **kwargs)
        slave.daemon = False
        return slave
      self.SlaveFactory = func
    else:
      self.QueueFactory = mp.Queue
      self.JoinableQueueFactory = mp.JoinableQueue
      self.SlaveFactory = mp.Process

  def split(self, list, nchunks=None, chunksize=None):
    """ Split every item in the list into nchunks, and return a list of chunked items.
           - then used with p.starmap(work, zip(*p.split((xxx,xxx,xxxx), chunksize=1024))
        For non sequence items and tuples, constructs a repeated iterator,
        For sequence items(but tuples), convert to numpy array then use nupy.array_split to split them.
        either give nchunks or chunksize. chunksize is only instructive, nchunk is estimated from chunksize
    """
    result = []
    if nchunks is None:
      if chunksize is None:
        nchunks = self.np * 2
      else:
        nchunks = 0
        for item in list:
          if hasattr(item, '__len__') and not isinstance(item, tuple):
            nchunks = int(len(item) / chunksize)
        if nchunks == 0: nchunks = 1
      
    for item in list:
      if isinstance(item, numpy.ndarray):
        result += [numpy.array_split(item, nchunks)]
      elif hasattr(item, '__getslice__') and not isinstance(item, tuple):
        result += [numpy.array_split(numpy.asarray(item), nchunks)]
      else:
        result += [repeat(item)]
    return result

  def starmap(self, work, sequence, chunksize=1, ordered=False):
    return self.map(work, sequence, chunksize, ordered=ordered, star=True)

  def map(self, work, sequence, chunksize=1, ordered=False, star=False):
    """
      calls work on every item in sequence. the return value is unordered unless ordered=True.
    """
    if not hasattr(sequence, '__getslice__'):
      raise TypeError('can only take a slicable sequence')

    def worker(S, sequence, Q):
      dead = False
      while True:
        begin, end = S.get()
        if begin is None and end is None: 
          S.task_done()
          break
        if dead: 
          S.task_done()
          continue
        out = []
        try:
          for i in sequence[begin:end]:
            if star: out += [ work(*i) ]
            else: out += [ work(i) ]
        except Exception as e:
          Q.put((e, traceback.format_exc()))
          dead = True
        finally:
          S.task_done()

        Q.put((begin, out))
      
    P = []
    Q = self.QueueFactory()
    S = self.JoinableQueueFactory()

    i = 0

    while i < len(sequence):
      j = i + chunksize 
      if j > len(sequence): j = len(sequence)
      S.put((i, j))
      i = j

    for i in range(self.np):
        S.put((None, None)) # sentinel
        p = self.SlaveFactory(target=worker, args=(S, sequence, Q))
        P.append(p)
        p.start()

    S.join()

#   the result is not sorted 
    R = []
    while not Q.empty():
      ind, r = Q.get()
      if isinstance(ind, Exception): 
        raise Exception(r)
      R += r
    
    return R

  def starmap_debug(self, work, sequence, chunksize=1):
    return self.map_debug(work, sequence, chunksize, star=True)

  def map_debug(self, work, sequence, chunksize=1, star=False):
    def worker(args):
       if star: return args[0](*(args[1]))
       else: return args[0](args[1])
    return map(worker, izip(cycle([work]), sequence))

# Pickling is needed only for mp.Pool. Our pool is directly based on Process
# thus no need to pickle anything

def __unpickle__(ai, dtype):
  dtype = numpy.dtype(dtype)
  tp = ctypeslib._typecodes['|u1']
  # if there are strides, use strides, otherwise the stride is the itemsize of dtype
  if ai['strides']:
    tp *= ai['strides'][-1]
  else:
    tp *= dtype.itemsize
  for i in numpy.asarray(ai['shape'])[::-1]:
    tp *= i
  # grab a flat char array at the sharemem address, with length at least contain ai required
  ra = tp.from_address(ai['data'][0])
  buffer = ctypeslib.as_array(ra).ravel()
  # view it as what it should look like
  shm = numpy.ndarray(buffer=buffer, dtype=dtype, 
      strides=ai['strides'], shape=ai['shape']).view(type=SharedMemArray)
  return shm

def __pickle__(obj):
  return obj.__reduce__()

class SharedMemArray(numpy.ndarray):
  """ 
      SharedMemArray works with multiprocessing.Pool through pickling.
      With sharedmem.Pool pickling is unnecssary. sharemem.Pool is recommended.

      Do not directly create an SharedMemArray or pass it to numpy.view.
      Use sharedmem.empty or sharedmem.copy instead.

      When a SharedMemArray is pickled, only the meta information is stored,
      So that when it is unpicled on the other process, the data is not copied,
      but simply viewed on the same address.
  """
  def __init__(self):
    pass
  def __reduce__(self):
    return __unpickle__, (self.__array_interface__, self.dtype)

copy_reg.pickle(SharedMemArray, __pickle__, __unpickle__)

def empty(shape, dtype='f8'):
  """ allocates an empty array on the shared memory """
  dtype = numpy.dtype(dtype)
  tp = ctypeslib._typecodes['|u1'] * dtype.itemsize
  ra = RawArray(tp, int(numpy.asarray(shape).prod()))
  shm = ctypeslib.as_array(ra)
  return shm.view(dtype=dtype, type=SharedMemArray).reshape(shape)

def copy(a):
  """ copies an array to the shared memory, use
     a = copy(a) to immediately dereference the old 'a' on private memory
   """
  shared = empty(a.shape, dtype=a.dtype)
  shared[:] = a[:]
  return shared

def wrap(a):
  return copy(a)

def argsort(data, chunksize=65536):
  """
     parallel argsort, like numpy.argsort

     first call numpy.argsort on nchunks of data,
     then merge the returned arg.
     it uses 2 * len(data) * int64.itemsize of memory during calculation,
     that is len(data) * int64.itemsize in addition to the size of the returned array.
  """

  from gaepsi.ccode import merge

  # round to power of two.
  nchunks = len(data) / chunksize
  if nchunks == 0: return data.argsort()

  if nchunks & (nchunks - 1) != 0: 
    v = nchunks - 1
    v |= v >> 1
    v |= v >> 2
    v |= v >> 4
    v |= v >> 8
    v |= v >> 16
    v |= v >> 32
    nchunks = v + 1

  arg1 = numpy.empty(len(data), dtype='i8')
  data_split = numpy.array_split(data, nchunks)
  sublengths = numpy.array([len(x) for x in data_split], dtype='i8')
  suboffsets = numpy.zeros(shape = sublengths.shape, dtype='i8')
  suboffsets[1:] = sublengths.cumsum()[:-1]

  arg_split = numpy.array_split(arg1, nchunks)

  with Pool(use_threads=True) as pool:
    def work(data, arg):
      arg[:] = data.argsort()
    pool.starmap(work, zip(data_split, arg_split))
  
  arg2 = numpy.empty(len(data), dtype='i8')

  while len(sublengths) > 1:
    with Pool(use_threads=True) as pool:
      def work(off1, len1, off2, len2, arg1, arg2, data):
        merge(data[off1:off1+len1+len2], arg1[off1:off1+len1], arg1[off2:off2+len2], arg2[off1:off1+len1+len2])

    pool.starmap(work, zip(suboffsets[::2], sublengths[::2], suboffsets[1::2], sublengths[1::2], repeat(arg1), repeat(arg2), repeat(data)))
    arg1, arg2 = arg2, arg1
    suboffsets = [x for x in suboffsets[::2]]
    sublengths = [x+y for x,y in zip(sublengths[::2], sublengths[1::2])]

  del arg2
  return arg1.view(type=numpy.ndarray)

