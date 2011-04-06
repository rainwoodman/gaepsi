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
parser.add_option("-z", "--zip", dest="zip", action="store_true", help="save to the gzip format instead of a direct dump(only the raster)")
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

parser.add_option("-G", "--gas", dest="gas", action="store_true", help="produce gas density xxx-gas.rst")
parser.add_option("-S", "--sfr", dest="sfr", action="store_true", help="produce gas mass weighted sfr sum xxx-msfr.rst") 
parser.add_option("-T", "--temp", dest="temp", action="store_true", help="process gas mass weighted temp sum(in KB/Mproton) xxx-mtemp.rst") 
parser.add_option("-S", "--star", dest="star", type="float", help="process star with the given circle size xxx-star.vec")
parser.add_option("-B", "--blackhole", dest="blackhole", type="float", help="process blackhole with the give circle size xxx-bh.vec")

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
stripe = Stripe(comm = comm, imagesize = opt.geometry,
                xcut = opt.xcut, ycut = opt.ycut, zcut = opt.zcut)

if opt.gas != None:
  gaslayer = RasterLayer(stripe, valuedtype='f4')
  gasfile = gaslayer.getfilename(opt.prefix, 'gas', '.rst');
if opt.sfr != None:
  msfrlayer = RasterLayer(stripe, valuedtype='f4')
  msfrfile = msfrlayer.getfilename(opt.prefix, 'msfr', '.rst')
if opt.temp != None:
  templayer = RasterLayer(stripe, valuedtype='f4')
  tempfile = templayer.getfilename(opt.prefix, 'mtemp', '.rst')
if opt.blackhole != None:
  bhlayer = ScatterLayer(stripe, valuedtype='f4', scale=opt.blackhole)
  bhfile = bhlayer.getfilename(opt.prefix, 'bh', '.vec')
if opt.star != None:
  starlayer = ScatterLayer(stripe, valuedtype='f4', scale=opt.star)
  starfile = starlayer.getfilename(opt.prefix, 'star', '.vec')
# when opt.range == None, there is no need to resume
# otherwise assuming the caller whats to resume if the starting point is not the
# first snapshot file.
if opt.range != None and opt.range[0] != 0:
  ses_reading = timer.session('reading unfinished data')
  if opt.gas != None:
    gaslayer.fromfile(gasfile, zip=opt.zip)
  if opt.sfr != None:
    msfrlayer.fromfile(msfrfile, zip=opt.zip)
  if opt.temp != None:
    templayer.fromfile(tempfile, zip=opt.zip)
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
  bhfield = Field(boxsize = opt.boxsize, components={'bhmass':'f4'})
  starfield = Field(boxsize = opt.boxsize, components={'sft':'f4'})
  gascomps = []
  if opt.sfr != None:
    gascomps += ['sfr']
  if opt.temp != None:
    gascomps += ['ie', 'reh']
  if opt.gas != None:
    gascomps += ['mass', 'sml']
  gasfield = Field(boxsize = opt.boxsize, components ={'mass':'f4', 'sml':'f4', 'ie':'f4', 'reh':'f4', 'sfr':'f4'})
  if opt.temp == None:
    del gasfield['reh']
    del gasfield['ie']
  if opt.sfr == None:
    del gasfield['sfr']
  if step < len(snaplist):
# the if above decides if a snapshot file is assigned to this core at this step
    snapfile = snaplist[step]
    snap = Snapshot(snapfile, opt.format)
    if opt.gas != None:
      gasfield.add_snapshot(snap, ptype=0, components=gascomps)
    if opt.sfr != None:
# replace the field sfr with mass-weighted value
      gasfield['sfr'][:] = gasfield['sfr'][:] * gasfield['mass'][:]
    if opt.temp != None:
      Xh = 0.76
      gasfield['ie'][:] = gasfield['mass'][:] * gasfield['ie'][:] / (gasfield['reh'][:] * Xh + (1 - Xh) * 0.25 + Xh) * (2.0 / 3.0)
    if opt.blackhole != None:
      bhfield.add_snapshot(snap, ptype=5, components=['bhmass'])
    if opt.star != None:
      starfield.add_snapshot(snap, ptype=4, components=['sft'])
    del snap
  ses_reading.end()

# then we rebalance the fields, moving the relavant particles to the correct cores

  ses_rebalance =  timer.session("rebalancing")
  if opt.blackhole != None:
    bhfield.unfold(opt.matrix.T)
    bhfield = stripe.rebalance(bhfield, bleeding = opt.blackhole)
    ses_rebalance.checkpoint("bh")
  if opt.star != None:
    starfield.unfold(opt.matrix.T)
    starfield = stripe.rebalance(starfield, bleeding = opt.star)
    ses_rebalance.checkpoint("star")
  if opt.gas != None:
    gasfield.unfold(opt.matrix.T)
    if opt.temp != None: 
      del gasfield['reh']
    gasfield = stripe.rebalance(gasfield)
    ses_rebalance.checkpoint("gas")

    load_all = array(comm.allgather(gasfield.numpoints * 1.0))
    if comm.rank == 0: print "load mean, max, penalty", load_all.mean(), load_all.max(), load_all.max() / load_all.mean()
    if comm.rank == 0:
      print 'gasfield boxsize', gasfield.boxsize
      print 'gasfield locations', gasfield.describe('locations')
  ses_rebalance.end()
# endif
# at this point the field is balanced over the cores
  ses_mklayers = timer.session("making layers")
  if opt.gas != None:
    layerlist = [gaslayer]
    gasvalues = ['mass']
    if opt.sfr != None:
      layerlist = layerlist + [msfrlayer]
      gasvalues = gasvalues + ['sfr']
    if opt.temp != None:
      layerlist = layerlist + [templayer]
      gasvalues = gasvalues + ['ie']
    stripe.mkraster(gasfield, gasvalues, layer=layerlist)
    numparticles = comm.allgather(gaslayer.numparticles)
    if comm.rank == 0: print "num gas particles per layer", numparticles
    ses_mklayers.checkpoint("gas layers")
  if opt.blackhole != None:
    stripe.mkscatter(bhfield, 'bhmass', scale=opt.blackhole, layer=bhlayer)
    numparticles = comm.allgather(bhlayer.numparticles)
    if comm.rank == 0: print "num bh particles per layer", numparticles
    ses_mklayers.checkpoint("bh")
  if opt.star != None:
    stripe.mkscatter(starfield, 'sft', scale=opt.star, layer=starlayer)
    numparticles = comm.allgather(starlayer.numparticles)
    if comm.rank == 0: print "num star particles per layer", numparticles
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
if opt.sfr != None:
  hist, bins = msfrlayer.hist()
  if comm.rank == 0:
    savez(opt.prefix + 'hist-msfr-sum.npz', bins = bins, hist=hist)
if opt.temp != None:
  hist, bins = templayer.hist()
  if comm.rank == 0:
    savez(opt.prefix + 'hist-mtemp-sum.npz' , bins = bins, hist=hist)

ses_hist.end()
#save
ses_writing = timer.session("writing output")
for i in range(opt.outputpatches):
  ses_patch = timer.session("writing output patch %d" % i)
  if opt.gas != None:
    if comm.rank % opt.outputpatches == i:
      gaslayer.tofile(gasfile, zip=opt.zip)
    ses_patch.checkpoint("gas")
  if opt.sfr != None:
    if comm.rank % opt.outputpatches == i:
      msfrlayer.tofile(msfrfile, zip=opt.zip)
      msfrlayer.data /= gaslayer.data
    ses_patch.checkpoint("sfr")
  if opt.temp != None:
    if comm.rank % opt.outputpatches == i:
      templayer.tofile(tempfile, zip=opt.zip)
      templayer.data /= gaslayer.data
    ses_patch.checkpoint("temp")
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

ses_hist = timer.session("histogram gas fields")
if opt.sfr != None:
  hist, bins = msfrlayer.hist()
  if comm.rank == 0:
    savez(opt.prefix + 'hist-msfr.npz', bins = bins, hist=hist)
if opt.temp != None:
  hist, bins = templayer.hist()
  if comm.rank == 0:
    savez(opt.prefix + 'hist-mtemp.npz', bins = bins, hist=hist)
ses_hist.end()
