import gadget
from gadget.plot.annotate import annotate
from gadget.remap import remap
from mpi4py import MPI
from numpy import zeros, array, empty, nonzero, float32
from pmkfield import pmkfield

def pannotate(comm, snapfile, imagesize, M, ptype, value):
  field, snap = pmkfield(comm, snapfile, M, ptype, values=[value])

  field['default'] = field[value]

  strip_px_start = imagesize[0] * comm.rank / (comm.size)
  strip_px_end = imagesize[0] * (comm.rank + 1) / (comm.size)
  npixels = [strip_px_end - strip_px_start, imagesize[1]]

  xrange = [strip_px_start * field.boxsize[0] / imagesize[0],
            strip_px_end   * field.boxsize[0] / imagesize[0]]
  yrange = [0, field.boxsize[1]]
  zrange = [0, field.boxsize[2]]
  return annotate(field, xrange = xrange, yrange = yrange, zrange = zrange, npixels = npixels)
