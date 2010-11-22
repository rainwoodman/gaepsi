from mpi4py import MPI
from numpy import array, zeros, empty, float32
from time import clock

from remap import remap
from snapshot import Snapshot
from field import Field
def pmkfield(comm, snapfile, M, ptype, values):
  """ make a field on all processes in comm from snapfile, unfold with 
      matrix M, based on particle type ptype, and loadin the blocks in
      the list values
   """
  if comm.rank == 0:
    print snapfile[0], snapfile[1]
    snap = Snapshot(snapfile[0], snapfile[1])
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

  field = Field(locations = newpos, boxsize=newboxsize, origin = zeros(3, 'f4'))

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
