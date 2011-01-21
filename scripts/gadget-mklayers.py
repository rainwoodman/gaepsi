#! python
from optparse import OptionParser, OptionValueError
from gadget.tools import parsearray, parsematrix
parser = OptionParser()
parser.add_option("-r", "--range", dest="range", type="string",
     action="callback", callback=parsearray, callback_kwargs=dict(sep=',', dtype='i4', len=2))
parser.add_option("-s", "--snapname", dest="snapname")
parser.add_option("-f", "--format", dest="format", help="the format of the snapshot")
parser.add_option("-b", "--boxsize", dest="boxsize", type="float", help="the boxsize of the snapshot")
parser.add_option("-g", "--geometry", dest="geometry",  type="string",
     action="callback", callback=parsearray, callback_kwargs=dict(sep='x', dtype='i4', len=2))
parser.add_option("-M", "--matrix", dest="matrix", type="string",
     action="callback", callback=parsematrix, callback_kwargs=dict(shape=(3,3)))
parser.add_option("-x", "--xcut", dest="xcut", type="string",
     action="callback", callback=parsearray, callback_kwargs=dict(sep=',', dtype='f4', len=2))
parser.add_option("-y", "--ycut", dest="ycut", type="string",
     action="callback", callback=parsearray, callback_kwargs=dict(sep=',', dtype='f4', len=2))
parser.add_option("-z", "--zcut", dest="zcut", type="string",
     action="callback", callback=parsearray, callback_kwargs=dict(sep=',', dtype='f4', len=2))
parser.add_option("-B", "--blackhole", dest="blackhole", type="float", help="process blackhole with the give circle size")
parser.add_option("-G", "--gas", dest="gas", type="string", help="process gas produce the given mass weighted field and mass field")
parser.add_option("-S", "--star", dest="star", type="float", help="process star with the given circle size")

opt, args = parser.parse_args()
if opt.boxsize==None:
  parser.error("specify --boxsize")
if opt.matrix == None:
  parser.error("specify --matrix")
if opt.geometry == None:
  parser.error("specify --geometry")
if opt.snapname == None:
  parser.error("specify --snapname")
if opt.format == None:
  parser.error("specify --format")

import sys
from numpy import max
from numpy import isnan
from mpi4py import MPI
comm = MPI.COMM_WORLD
if comm.rank == 0: print opt
from gadget.tools import timer

comm.Barrier()
if comm.rank == 0: print 'program started', timer.reset()
from gadget.tools.stripe import *
from gadget.remap import remap
from numpy import array
comm.Barrier()
if comm.rank == 0: print 'done loading modules', timer.restart()

# create the stripes and layers
stripe = Stripe(comm = comm, imagesize = opt.geometry, boxsize = remap(opt.matrix.T) * opt.boxsize,
                xcut = opt.xcut, ycut = opt.ycut, zcut = opt.zcut)

if opt.gas != None:
  gaslayer = RasterLayer(stripe, valuedtype='f4')
  msfrlayer = RasterLayer(stripe, valuedtype='f4')
if opt.blackhole != None:
  bhlayer = VectorLayer(stripe, valuedtype='f4', scale=opt.blackhole)
if opt.star != None:
  starlayer = VectorLayer(stripe, valuedtype='f4', scale=opt.star)


# when opt.range == None, there is only one file in the snapshot
if opt.range != None:
  snapfiles = [opt.snapname % i for i in range(opt.range[0], opt.range[1])]
else:
  snapfiles = [opt.snapname]

# when opt.range == None, there is no need to resume
# otherwise assuming the caller whats to resume if the starting point is not the
# first snapshot file.
if opt.range != None and opt.range[0] != 0:
  comm.Barrier()
  if comm.rank == 0: print 'start reading', timer.restart()
  if opt.gas != None:
    gaslayer.fromfile('%03d-gas.rst' % comm.rank)
    msfrlayer.fromfile('%03d-msfr.rst' % comm.rank)
  if opt.blackhole != None:
    bhlayer.fromfile('%03d-bh.vec' % comm.rank)
  if opt.star != None:
    starlayer.fromfile('%03d-star.vec' % comm.rank)
  comm.Barrier()
  if comm.rank == 0: print 'done reading', timer.restart()

# process the field snapshot files

snaplist_all = [
    snapfiles[len(snapfiles) * rank / comm.size: len(snapfiles) * (rank+1) / comm.size]
    for rank in range(comm.size)]
# steps is the number of iterations to scan over all snapshot files.
# in each step each core loads in one file and an alltoall communication
# distributes relevant particles to the core hosting the stripe
steps = max([len(snaplist) for snaplist in snaplist_all])
#snaplist on this core
snaplist = snaplist_all[comm.rank]
for step in range(steps):
# the fields may be None if the core doesn't read in a file in this step
# it happens at the end of the iterations, when NCPU doesn't divide the
# number of snapshot files given by opt.range
  bhfield = None
  starfield = None
  gasfield = None
  if comm.rank == 0: print "preparing fields step", step, timer.restart()
  if step < len(snaplist):
# the if above decides if a snapshot file is assigned to this core at this step
    snapfile = snaplist[step]
    snap = Snapshot(snapfile, opt.format)
    if opt.gas != None:
      gasfield = Field(snap = snap, ptype=0, values=['mass', 'sml', opt.gas])
      gasfield.unfold(opt.matrix.T)
    if opt.blackhole != None:
      bhfield = Field(snap = snap, ptype=5, values=['bhmass'])
      bhfield.unfold(opt.matrix.T)
    if opt.star != None:
      starfield = Field(snap = snap, ptype=4, values=['mass'])
      starfield.unfold(opt.matrix.T)
# endif
  comm.Barrier()
# cores assigned with snapshot files now have a snapshot loaded into a Field
# cores unassigned have Nones 
  if comm.rank == 0: print "start rebalancing", timer.restart()
  if opt.blackhole != None:
    bhfield = stripe.rebalance(bhfield, values=['bhmass'], bleeding = opt.blackhole)
    comm.Barrier()
    if comm.rank == 0: 
      print "bh balanced", timer.restart()
  if opt.star != None:
    starfield = stripe.rebalance(starfield, values=['mass'], bleeding = opt.star)
    comm.Barrier()
    if comm.rank == 0: print "star balanced", timer.restart()
  if opt.gas != None:
    gasfield = stripe.rebalance(gasfield, values=['sml', 'mass', opt.gas])
    comm.Barrier()
    if comm.rank == 0: print "gas balanced", timer.restart()
    load_all = array(comm.allgather(gasfield.numpoints * 1.0))
    if comm.rank == 0: print "load", load_all / load_all.max()
    if comm.rank == 0: print "load mean, max, penalty", load_all.mean(), load_all.max(), load_all.max() / load_all.mean()
    if comm.rank == 0: print "fields prepared", timer.restart()
# replace the field 'opt.gas' with mass-weighted value
    gasfield[opt.gas] = gasfield[opt.gas] * gasfield['mass']

    if comm.rank == 0:
      print 'gasfield boxsize', gasfield.boxsize
      print 'gasfield locations', gasfield.describe('locations')
# endif
# at this point the field is balanced over the cores
  if comm.rank == 0: print 'making layers', timer.restart()
  if opt.gas != None:
    stripe.mkraster(gasfield, ['mass', opt.gas], layer=[gaslayer, msfrlayer])
    comm.Barrier()
    if comm.rank == 0: print 'gas and sfr rasterized', timer.restart()
  if opt.blackhole != None:
    stripe.mkvector(bhfield, 'bhmass', scale=opt.blackhole, layer=bhlayer)
    comm.Barrier()
    if comm.rank == 0: print 'bh vectorized', timer.restart()
  if opt.star != None:
    stripe.mkvector(starfield, 'mass', scale=opt.star, layer=starlayer)
    comm.Barrier()
    if comm.rank == 0: print 'star vectorized', timer.restart()

#save 'n' reload
comm.Barrier()
if comm.rank == 0: print 'start writing', timer.restart()
if opt.blackhole != None:
  bhlayer.tofile('%03d-bh.vec' % comm.rank)
if opt.gas != None:
  gaslayer.tofile('%03d-gas.rst' % comm.rank)
  msfrlayer.tofile('%03d-msfr.rst' % comm.rank)
if opt.star != None:
  starlayer.tofile('%03d-star.vec' % comm.rank)
comm.Barrier()
if comm.rank == 0: print 'done writeing', timer.restart()

#M = matrix([[2,1,0],[0,2,1],[1,0,0]]).T
#imagesize = array([286, 262])

