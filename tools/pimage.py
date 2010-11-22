import numpy
from mpi4py import MPI
from time import clock
from numpy import array, zeros, empty, float32, fromfile

from gadget.plot.image import image
from gadget.remap import remap
from gadget.snapshot import Snapshot
from gadget.field import Field

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

def mkimage(stripe, snapname, format, FIDS, M, ptype, fieldname=None):
  values = ['mass', 'sml']
  if fieldname != None:
    values = values + [fieldname]

  if FIDS != None:
    for fid in FIDS:
      field = mkfield(stripe.comm, snapname % fid, format, M=M, ptype=0, values=values)
      if fieldname==None:
        field['default'] = field['mass']
      else:
        field['default'] = field[fieldname] * field['mass']
      stripe.add(field)
  else:
    field = mkfield(stripe.comm, snapname, format, M=M, ptype=0, values=values)
    if fieldname==None:
      field['default'] = field['mass']
    else:
      field['default'] = field[fieldname] * field['mass']
    stripe.add(field)

  min, max, sum = stripe.stat()
  if stripe.comm.rank == 0:
    print "max = ", max, "min = ", min, "stat = ", sum

def mkfield(comm, snapname, format, M, ptype, values):
  """ make a field on all processes in comm from snapfile, unfold with 
      matrix M, based on particle type ptype, and loadin the blocks in
      the list values
   """
  if comm.rank == 0:
    print snapname, format
    snap = Snapshot(snapname, format, mode='r', buffering=1048576*32)
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
    print "constructing field(pos", values, ")", clock()
    snap.load(['pos'], ptype = ptype)
    pos = snap.P[ptype]['pos']
    snap.load(values, ptype = ptype)
  else :
    pos = empty(dtype='f4', shape = 0)

  comm.Barrier()
  if comm.rank == 0: print 'start unfold', clock()

  newpos, newboxsize = premap(comm, M, pos, N[ptype], boxsize)

  if comm.rank == 0: print 'newboxsize =', newboxsize
  if comm.rank == 0: print 'pos', newpos.max(axis=0), newpos.min(axis=0)
  if comm.rank == 0: print 'done unfold', clock()

  comm.Barrier()

  field = Field(locations = newpos, boxsize=newboxsize, origin = zeros(3, 'f4'))

  for value in values:
    if comm.rank == 0:
      v = snap.P[ptype][value]
    else :
      v = empty(dtype='f4', shape = N[ptype])
    comm.Bcast(v)
    field[value] = v     

  if comm.rank == 0: print 'done construction', clock()
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

