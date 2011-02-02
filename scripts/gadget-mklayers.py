#! python
from gadget.tools import timer
ses_import = timer.session('import extra modules')
from optparse import OptionParser, OptionValueError
from gadget.tools.cmdline import parsearray, parsematrix
parser = OptionParser(conflict_handler="resolve")
parser.add_option("-r", "--range", dest="range", type="string",
     action="callback", callback=parsearray, callback_kwargs=dict(sep=',', dtype='i4', len=2))
parser.add_option("-r", "--rangefile", dest="rangefile", type="string", help="a file listing file ids to operate")
parser.add_option("-s", "--snapname", dest="snapname")
parser.add_option("-f", "--format", dest="format", help="the format of the snapshot")
parser.add_option("-b", "--boxsize", dest="boxsize", type="float", help="the boxsize of the snapshot; unused(decided from snapfile)")
parser.add_option("-g", "--geometry", dest="geometry",  type="string",
     action="callback", callback=parsearray, callback_kwargs=dict(sep='x', dtype='i4', len=2))
parser.add_option("-I", "--inputpatches", dest="inputpatches",  type="int", default=1, help="number of patches to divided the input(snapfiles only) into")
parser.add_option("-O", "--outputpatches", dest="outputpatches",  type="int", default=1, help="number of patches to divided the output into")
parser.add_option("-p", "--prefix", dest="prefix", type="string", default='', help="prefix of the outputs")
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
if opt.matrix == None:
  parser.error("specify --matrix")
if opt.geometry == None:
  parser.error("specify --geometry")
if opt.snapname == None:
  parser.error("specify --snapname")
if opt.format == None:
  parser.error("specify --format")

from gadget.tools import _MPI as MPI

comm = MPI.COMM_WORLD
if comm.rank == 0: print opt

from gadget.tools.stripe import *
from gadget.snapshot import Snapshot
from gadget.field import Field
from gadget.remap import remap
from numpy import array
ses_import.end()

# when opt.range == None, there is only one file in the snapshot
if opt.range != None:
  snapfiles = [opt.snapname % i for i in range(opt.range[0], opt.range[1])]
else:
  if opt.rangefile !=None:
    from numpy import loadtxt
    snapids = loadtxt(opt.rangefile)
    snapfiles = [opt.snapname % i for i in snapids]
  else:
    snapfiles = [opt.snapname]

#decide the boxsize
if comm.rank == 0:
  snap = Snapshot(snapfiles[0], opt.format)
  boxsize = snap.C['L']
  del snap
else: boxsize = None
opt.boxsize = comm.bcast(boxsize)
del boxsize

# create the stripes and layers
stripe = Stripe(comm = comm, imagesize = opt.geometry, boxsize = remap(opt.matrix.T) * opt.boxsize,
                xcut = opt.xcut, ycut = opt.ycut, zcut = opt.zcut)

if opt.gas != None:
  gaslayer = RasterLayer(stripe, valuedtype='f4')
  msfrlayer = RasterLayer(stripe, valuedtype='f4')
  gasfile = gaslayer.getfilename(opt.prefix, 'gas', '.rst');
  msfrfile = msfrlayer.getfilename(opt.prefix, 'm' + opt.gas, '.rst')
if opt.blackhole != None:
  bhlayer = VectorLayer(stripe, valuedtype='f4', scale=opt.blackhole)
  bhfile = bhlayer.getfilename(opt.prefix, 'bh', '.vec')
if opt.star != None:
  starlayer = VectorLayer(stripe, valuedtype='f4', scale=opt.star)
  starfile = starlayer.getfilename(opt.prefix, 'star', '.vec')
# when opt.range == None, there is no need to resume
# otherwise assuming the caller whats to resume if the starting point is not the
# first snapshot file.
if opt.range != None and opt.range[0] != 0:
  ses_reading = timer.session('reading unfinished data')
  if opt.gas != None:
    gaslayer.fromfile(gasfile)
    msfrlayer.fromfile(msfrfile)
  if opt.blackhole != None:
    bhlayer.fromfile(bhfile)
  if opt.star != None:
    starlayer.fromfile(starfile)
  ses_reading.end()

# process the field snapshot files
Nreaders = comm.size / opt.inputpatches

snaplist_per_reader = [
    snapfiles[len(snapfiles) * rank / Nreaders: len(snapfiles) * (rank+1) / Nreaders]
    for rank in range(Nreaders)]
snaplist_all = [[]] * comm.size
for rank in range(Nreaders):
  snaplist_all[rank * opt.inputpatches] = snaplist_per_reader[rank]

# steps is the number of iterations to scan over all snapshot files.
# in each step each core loads in one file and an alltoall communication
# distributes relevant particles to the core hosting the stripe
steps = max([len(snaplist) for snaplist in snaplist_all])
#snaplist on this core
snaplist = snaplist_all[comm.rank]

# start the main loop
for step in range(steps):
# the fields may be None if the core doesn't read in a file in this step
# it happens at the end of the iterations, when NCPU doesn't divide the
# number of snapshot files given by opt.range
  ses_step = timer.session("step %d" % step)
  ses_reading = timer.session('reading fields')
  bhfield = None
  starfield = None
  gasfield = None
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
    del snap
  ses_reading.end()

# cores assigned with snapshot files now have a snapshot loaded into a Field
# cores unassigned have Nones 
  ses_rebalance =  timer.session("rebalancing")
  if opt.blackhole != None:
    bhfield = stripe.rebalance(bhfield, values=['bhmass'], bleeding = opt.blackhole)
    ses_rebalance.checkpoint("bh")
  if opt.star != None:
    starfield = stripe.rebalance(starfield, values=['mass'], bleeding = opt.star)
    ses_rebalance.checkpoint("star")
  if opt.gas != None:
    gasfield = stripe.rebalance(gasfield, values=['sml', 'mass', opt.gas])
    ses_rebalance.checkpoint("gas")

    load_all = array(comm.allgather(gasfield.numpoints * 1.0))
    if comm.rank == 0: print "load", load_all / load_all.max()
    if comm.rank == 0: print "load mean, max, penalty", load_all.mean(), load_all.max(), load_all.max() / load_all.mean()
# replace the field 'opt.gas' with mass-weighted value
    gasfield[opt.gas][:] = gasfield[opt.gas][:] * gasfield['mass'][:]
    if comm.rank == 0:
      print 'gasfield boxsize', gasfield.boxsize
      print 'gasfield locations', gasfield.describe('locations')
  ses_rebalance.end()
# endif
# at this point the field is balanced over the cores
  ses_mklayers = timer.session("making layers")
  if opt.gas != None:
    stripe.mkraster(gasfield, ['mass', opt.gas], layer=[gaslayer, msfrlayer])
    ses_mklayers.checkpoint("gas and sfr")
  if opt.blackhole != None:
    stripe.mkvector(bhfield, 'bhmass', scale=opt.blackhole, layer=bhlayer)
    ses_mklayers.checkpoint("bh")
  if opt.star != None:
    stripe.mkvector(starfield, 'mass', scale=opt.star, layer=starlayer)
    ses_mklayers.checkpoint("star")
  ses_mklayers.end()
  ses_step.end()
#end of the main loop
del starfield
del bhfield
del gasfield
ses_hist = timer.session("histogram")
from numpy import savez
if opt.blackhole != None:
  hist, bins = bhlayer.hist()
  if comm.rank == 0:
    savez(opt.prefix + 'hist-bh.npz', bins = bins, hist=hist)
if opt.star != None:
  hist, bins = starlayer.hist()
  if comm.rank == 0:
    savez(opt.prefix + 'hist-star.npz', bins = bins, hist=hist)
if opt.gas != None:
  hist, bins = gaslayer.hist()
  if comm.rank == 0:
    savez(opt.prefix + 'hist-gas.npz', bins = bins, hist=hist)
  hist, bins = msfrlayer.hist()
  if comm.rank == 0:
    savez(opt.prefix + 'hist-m%s-sum.npz' % opt.gas, bins = bins, hist=hist)

ses_hist.end()
#save
ses_writing = timer.session("writing output")
for i in range(opt.outputpatches):
  ses_patch = timer.session("writing output patch %d" % i)
  if opt.gas != None:
    if comm.rank % opt.outputpatches == i:
      gaslayer.tofile(gasfile)
    ses_patch.checkpoint("gas")
    if comm.rank % opt.outputpatches == i:
      msfrlayer.tofile(msfrfile)
      msfrlayer.data /= gaslayer.data
      del gaslayer
    ses_patch.checkpoint("sfr")
  if opt.blackhole != None:
    if comm.rank % opt.outputpatches == i:
      bhlayer.tofile(bhfile)
    ses_patch.checkpoint("bh")
  if opt.star != None:
    if comm.rank % opt.outputpatches == i:
      starlayer.tofile(starfile)
    ses_patch.checkpoint("star")
  ses_patch.end()
ses_writing.end()

ses_hist = timer.session("histogram msfr")
if opt.gas != None:
  hist, bins = msfrlayer.hist()
  if comm.rank == 0:
    savez(opt.prefix + 'hist-m%s.npz' % opt.gas, bins = bins, hist=hist)
ses_hist.end()
