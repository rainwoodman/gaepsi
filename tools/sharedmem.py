from multiprocessing.sharedctypes import RawArray
from itertools import cycle, izip
import multiprocessing as mp
import ctypes
import traceback
import copy_reg
from numpy import ctypeslib
import numpy
from multiprocessing import pool


class Pool:
  def __enter__(self):
    return self
  def __exit__(self, type, value, traceback):
    pass

  def __init__(self, np=None):
    if np is None: np = mp.cpu_count()
    self.np = np

  def starmap(self, work, iterable, chunksize=1):
    def worker(S, iterable, Q):
      dead = False
      while True:
        begin, end = S.get()
        if dead: 
          S.task_done()
          continue
        out = []
        try:
          for i in iterable[begin:end]:
            out += [ work(*i) ]
        except Exception as e:
          Q.put((e, traceback.format_exc()))
          dead = True
        finally:
          S.task_done()

        Q.put((begin, out))
      
    P = []
    Q = mp.Queue()
    S = mp.JoinableQueue()

    i = 0
    while i < len(iterable):
      j = i + chunksize 
      if j > len(iterable): j = len(iterable)
      S.put((i, j))
      i = j

    for i in range(self.np):
        p = mp.Process(
                target=worker,
                args=(S, iterable, Q))
        P.append(p)
        p.start()

    S.join()

    for p in P:
      p.terminate()

#   the result is not sorted 
    R = []
    while not Q.empty():
      ind, r = Q.get()
      if isinstance(ind, Exception): 
        raise Exception(r)
      R += r
    
    return R

  def starmap_debug(self, work, iterable, chunksize=1):
    def worker(args):
       args[0](*(args[1]))
    return map(worker, izip(cycle([work]), iterable))

# Pickling is needed only for mp.Pool. Out pool is directly based on Process
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
  def __init__(self):
    pass
  def __reduce__(self):
    return __unpickle__, (self.__array_interface__, self.dtype)

copy_reg.pickle(SharedMemArray, __pickle__, __unpickle__)

def empty(shape, dtype='f8'):
  dtype = numpy.dtype(dtype)
  tp = ctypeslib._typecodes['|u1'] * dtype.itemsize
  ra = RawArray(tp, numpy.asarray(shape).prod())
  shm = ctypeslib.as_array(ra)
  return shm.view(dtype=dtype, type=SharedMemArray).reshape(shape)

def wrap(a):
  shared = empty(a.shape, dtype=a.dtype)
  shared[:] = a[:]
  return shared

