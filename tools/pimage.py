import gadget
from gadget.plot.image import image
from gadget.remap import remap
from mpi4py import MPI
from time import clock
from numpy import array, zeros, empty, float32


def pimage(comm, snapfile, imagesize, M):
  """ parallelly do unfolding with matrix M and image an snapshot file into imagesize. The returned array is the image local stored on the process. snapfile is a tuple (filename, reader) (reader ='hydro3200', for example)"""
  strip_px_start = imagesize[0] * comm.rank / comm.size
  strip_px_end = imagesize[0] * (comm.rank + 1) / comm.size
  npixels = [strip_px_end - strip_px_start, imagesize[1]]
  if snapfile == None: # create a zero image
    return zeros(dtype='f4', shape=npixels)

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
    snap.load(['pos', 'sml', 'mass'], ptype = 0)
    pos = snap.P[0]['pos']
    sml = snap.P[0]['sml']
    value = snap.P[0]['mass']
  else :
    sml = empty(dtype='f4', shape = N[0])
    value = empty(dtype='f4', shape = N[0])
    pos = empty(dtype='f4', shape = 0)

  newpos = empty(dtype=('f4', 3), shape = N[0])

  if comm.rank == 0: print 'start unfold', clock()
  bufsize = N[0] / comm.size * 3
  local_pos = empty(shape = bufsize / 3, dtype=('f4', 3))
  comm.Scatter([pos, bufsize, MPI.FLOAT], 
               [local_pos, bufsize, MPI.FLOAT], root = 0)
  local_pos /= boxsize
  local_newpos, newboxsize = remap(M, local_pos)
  local_newpos *= boxsize
  comm.Allgather([local_newpos, bufsize, MPI.FLOAT], 
                 [newpos, bufsize, MPI.FLOAT])
  newboxsize = float32(newboxsize * boxsize)
  if comm.rank == 0: print 'newboxsize =', newboxsize
  if comm.rank == 0: print 'pos', newpos.max(axis=0), newpos.min(axis=0)
  if comm.rank == 0: print 'done unfold', clock()

  comm.Barrier()

  if comm.rank == 0: print 'start comm', clock()
  comm.Bcast([sml, MPI.FLOAT], 0)
  comm.Bcast([value, MPI.FLOAT], 0)
  comm.Barrier()
  if comm.rank == 0: print 'end comm', clock()

  field = gadget.Field(locations = newpos, boxsize=newboxsize, origin = zeros(3, 'f4'), ptype = 0)

  field['sml'] = sml
  field['default'] = value

  xrange = [strip_px_start * field.boxsize[0] / imagesize[0],
            strip_px_end   * field.boxsize[0] / imagesize[0]]
  yrange = [0, field.boxsize[1]]
  zrange = [0, field.boxsize[2]]
  if comm.rank == 0: print 'start image', clock()
  tmpim = image(field, xrange = xrange, yrange = yrange, zrange = zrange, npixels = npixels)
  comm.Barrier()
  if comm.rank == 0: print 'done image', clock()

  max = tmpim.max()
  min = tmpim.min()

  if max > 1000 or min < 0:
    print 'bad values in im', 
    print 'fid = ', snapfile, 'rank = ', comm.rank
    print 'xyzrange =', xrange, yrange, zrange
    print 'value = ', value.min(), value.max()
    print 'sml = ', sml.min(), sml.max()
    print 'pos = ', newpos.min(axis=0), newpos.max(axis=0)
    print 'boxsize = ', field.boxsize
    print 'origin = ', field.origin
    print 'num of violations: min', sum(tmpim.ravel() < 0)
    print 'num of violations: max', sum(tmpim.ravel() > 1000)
  
  return tmpim

def pminmax(comm, im): 
  maxsend = array([im.max()], 'f4')
  minsend = array([im.min()], 'f4')
  maxrecv = array([0], 'f4')
  minrecv = array([0], 'f4')

#  print 'maxsend = ', maxsend, 'maxrecv = ', maxrecv, comm.rank
  comm.Allreduce([maxsend, MPI.FLOAT], [maxrecv, MPI.FLOAT], op = MPI.MAX)
  comm.Allreduce([minsend, MPI.FLOAT], [minrecv, MPI.FLOAT], op = MPI.MIN)
  return minrecv, maxrecv
  
