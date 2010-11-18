import gadget
import numpy
from gadget.plot.image import image
from gadget.remap import remap
from mpi4py import MPI
from time import clock
from numpy import array, zeros, empty, float32, fromfile

class Stripe():
  def __init__(self, comm, imagesize, dtype='f4'):
    strip_px_start = imagesize[0] * comm.rank / comm.size
    strip_px_end = imagesize[0] * (comm.rank + 1) / comm.size
    self.npixels = [strip_px_end - strip_px_start, imagesize[1]]
  
    self.image = zeros(dtype=dtype, shape = self.npixels)
    self.px_start = strip_px_start
    self.px_end = strip_px_end
    self.imagesize = imagesize
    self.comm = comm
    self.dtype = dtype
  def add(self, field):
    """ parallelly do unfolding with matrix M and image an snapshot file into imagesize. The returned array is the image local stored on the process. snapfile is a tuple (filename, reader) (reader ='hydro3200', for example)"""

    xrange = [self.px_start * field.boxsize[0] / self.imagesize[0],
              self.px_end * field.boxsize[0] / self.imagesize[0]]
    yrange = [0, field.boxsize[1]]
    zrange = [0, field.boxsize[2]]
    if self.comm.rank == 0: print 'start image', clock()
    image(field, xrange = xrange, yrange = yrange, zrange = zrange, npixels=self.npixels, quick=False, target=self.image)
    self.comm.Barrier()
    if self.comm.rank == 0: print 'done image', clock()

  def stat(self):
    """ returns the min, max, and sum of the image across all processors"""
    maxsend = self.image.max()
    minsend = self.image.min()
    sumsend = self.image.sum()
    maxrecv = self.comm.allreduce(maxsend, op = MPI.MAX)
    minrecv = self.comm.allreduce(minsend, op = MPI.MAX)
    sumrecv = self.comm.allreduce(sumsend, op = MPI.SUM)
    return minrecv, maxrecv, sumrecv

  def tofile(self, file, dtype='f4'):
    if dtype != self.dtype:
      numpy.dtype(dtype).type(self.image).tofile(file)
    else:
      self.image.tofile(file)

  def fromfile(self, file, dtype='f4'):
    if dtype != self.dtype:
      self.image = numpy.dtype(self.dtype).type(fromfile(file, dtype=dtype))
    else:
      self.image = fromfile(file, dtype=dtype)
    self.image.shape = self.npixels[0], self.npixels[1]

def mkfield(comm, snapfile, M, ptype, values):
  """ make a field on all processes in comm from snapfile, unfold with 
      matrix M, based on particle type ptype, and loadin the blocks in
      the list values
   """
  if comm.rank == 0:
    print snapfile[0], snapfile[1]
    snap = gadget.Snapshot(snapfile[0], snapfile[1])
    N = snap.N.copy()
    boxsize = array([snap.C['L']], dtype='f4')
    print "N = ", N
  else :
    N = zeros(6, dtype='u4')
    boxsize = zeros(1, dtype='f4')
  
  comm.Barrier()
  comm.Bcast([N, MPI.INT], root = 0)
  comm.Bcast([boxsize, MPI.FLOAT], root = 0)

  if comm.rank == 0:
    print "start reading", clock()
    snap.load(['pos'], ptype = ptype)
    pos = snap.P[ptype]['pos']
  else :
    pos = empty(dtype='f4', shape = 0)

  if comm.rank == 0: print 'start unfold', clock()

  newpos, newboxsize = premap(comm, M, pos, N[ptype], boxsize)

  if comm.rank == 0: print 'newboxsize =', newboxsize
  if comm.rank == 0: print 'pos', newpos.max(axis=0), newpos.min(axis=0)
  if comm.rank == 0: print 'done unfold', clock()

  comm.Barrier()

  field = gadget.Field(locations = newpos, boxsize=newboxsize, origin = zeros(3, 'f4'))

  if comm.rank == 0:
    snap.load(values, ptype = ptype)
  for value in values:
    if comm.rank == 0:
      v = snap.P[ptype][value]
    else :
      v = empty(dtype='f4', shape = N[ptype])
    comm.Bcast([v, MPI.FLOAT], 0)
    field[value] = v     

  return field

def premap(comm, M, pos, N, boxsize):
  newpos = empty(dtype=('f4', 3), shape = N)
  bufsize = N / comm.size
  tail = bufsize * comm.size

  if N == 0:
    e, newboxsize = remap(M, pos)
  
  if bufsize > 0 :
    local_pos = empty(shape = bufsize, dtype=('f4', 3))
    comm.Scatter([pos, bufsize * 3, MPI.FLOAT], 
                 [local_pos, bufsize * 3, MPI.FLOAT], root = 0)
  
    local_pos /= boxsize
    local_newpos, newboxsize = remap(M, local_pos)
    local_newpos *= boxsize
    comm.Allgather([local_newpos, bufsize * 3, MPI.FLOAT], 
                   [newpos, bufsize * 3, MPI.FLOAT])
    newboxsize = float32(newboxsize * boxsize)
  if tail < N:
    if comm.rank == 0:
      local_pos = pos[tail:, :]
      local_pos /= boxsize
      leftover, newboxsize = remap(M, local_pos)
      leftover *= boxsize
      newboxsize = float32(newboxsize * boxsize)
    else:
      leftover = empty(dtype=('f4', 3), shape = N - tail)
      newboxsize = empty(dtype='f4', shape = 3)

    comm.Bcast([leftover, 3 * (N - tail), MPI.FLOAT], root = 0)
    comm.Bcast([newboxsize,MPI.FLOAT], root = 0)
    newpos[tail:, :] = leftover

  return newpos, newboxsize
