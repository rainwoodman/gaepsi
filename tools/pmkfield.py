from mpi4py import MPI
from numpy import array, zeros, empty, float32
from time import clock

from gadget.remap import remap
from gadget.snapshot import Snapshot
from gadget.field import Field

def mkfield(stripe, snapnames, format, ptype, values):
  "make a field that is locally relevant to the stripe"
  snap = Snapshot(snapnames[stripe.comm.rank], format)
  field = Field(snap = snap, ptype = ptype, values = values)
  field.unfold(M)
  argsort = field['locations'][:, 0].argsort()
  field['locations'] = field['locations'][argsort, :]
  for v in values:
    field[v] = field[v][argsort]
  
