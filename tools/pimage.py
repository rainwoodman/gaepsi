from mpi4py import MPI
from time import clock
from numpy import array, zeros, empty, float32

from gadget.remap import remap
from gadget.snapshot import Snapshot
from gadget.field import Field

def mkfield(comm, snapname, format, ptype, values):
  """ make a field on all processes in comm from snapfile, unfold with 
      matrix M, based on particle type ptype, and loadin the blocks in
      the list values
   """
  if comm.rank == 0:
    snap = Snapshot(snapname, format, mode='r', buffering=1048576*32)
    N = snap.N.copy()
    boxsize = array([snap.C['L']], dtype='f4')
    print N
  else :
    N = zeros(6, dtype='u4')
    boxsize = zeros(1, dtype='f4')
  
  comm.Barrier()
  comm.Bcast([N, MPI.INT], root = 0)
  comm.Bcast([boxsize, MPI.FLOAT], root = 0)

  if comm.rank == 0:
    snap.load(['pos'], ptype = ptype)
    pos = snap.P[ptype]['pos']
    snap.load(values, ptype = ptype)
  else:
    pos = empty(N[ptype], dtype = ('f4', 3))

  field = Field(locations = pos, boxsize=boxsize, origin = zeros(3, 'f4'))

  for value in values:
    if comm.rank == 0:
      v = snap.P[ptype][value]
    else :
      v = empty(dtype='f4', shape = N[ptype])
    comm.Bcast(v)
    field[value] = v

  comm.Barrier()
  return field

def unfold(comm, field, M):
  N = field['locations'].shape[0]
  if comm.rank == 0:
    oldpos = field['locations'].copy()
  else:
    oldpos = empty(dtype=('f4', 3), shape = 0)
  newpos = field['locations'].view()

  comm.Barrier()
  displs = [ i * N / comm.size * 3 for i in range(comm.size)]
  counts = [ (i+1) * N / comm.size * 3 - i * N / comm.size * 3for i in range(comm.size)]

  local_pos = empty(shape = counts[comm.rank], dtype='f4')
  comm.Scatterv([oldpos, counts, displs, MPI.FLOAT], 
                 [local_pos, MPI.FLOAT], root = 0)
  
  local_pos.shape = -1, 3
  local_pos /= field.boxsize
  local_newpos, newboxsize = remap(M, local_pos)
  local_newpos *= field.boxsize
  local_pos.shape = -1
  comm.Allgatherv([local_newpos, MPI.FLOAT], 
                 [newpos, counts, displs, MPI.FLOAT])
  newboxsize = float32(newboxsize * field.boxsize)

  comm.Barrier()
  field.boxsize = newboxsize
  comm.Barrier()

