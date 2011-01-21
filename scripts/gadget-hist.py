#! python
from optparse import OptionParser, OptionValueError
from gadget.tools import parsearray, parsematrix
parser = OptionParser()
parser.add_option("-g", "--geometry", dest="geometry",  type="string",
     action="callback", callback=parsearray, callback_kwargs=dict(sep='x', dtype='i4', len=2))
parser.add_option("-B", "--blackhole", dest="blackhole", type="float", help="process blackhole with the give circle size")
parser.add_option("-G", "--gas", dest="gas", type="string", help="process gas produce the given mass weighted field and mass field")
parser.add_option("-S", "--star", dest="star", type="float", help="process star with the given circle size")

opt, args = parser.parse_args()
if opt.geometry == None:
  parser.error("specify --geometry")

from gadget.tools import timer
from gadget.tools.stripe import *
from matplotlib.pyplot import plot, clf, title, savefig
from numpy import save
def plothist(comm, filename, hist):
  if comm.rank != 0: return
  clf()
  plot(linspace(0, 1, len(hist)), log10(hist))
  title(filename)
  savefig(filename)

from mpi4py import MPI
comm = MPI.COMM_WORLD

stripe = Stripe(comm = comm, imagesize = opt.geometry) 

if opt.gas != None:
  gaslayer = RasterLayer(stripe, valuedtype='f4')
  msfrlayer = RasterLayer(stripe, valuedtype='f4')
if opt.blackhole != None:
  bhlayer = VectorLayer(stripe, valuedtype='f4', scale=opt.blackhole)
if opt.star != None:
  starlayer = VectorLayer(stripe, valuedtype='f4', scale=opt.star)

comm.Barrier()
if comm.rank == 0: print 'start reading', timer.restart()
if opt.gas != None:
  gaslayer.fromfile('%03d-gas.rst' % comm.rank)
  msfrlayer.fromfile('%03d-msfr.rst' % comm.rank)
  msfrlayer.data /= gaslayer.data
if opt.blackhole != None:
  bhlayer.fromfile('%03d-bh.vec' % comm.rank)
if opt.star != None:
  starlayer.fromfile('%03d-star.vec' % comm.rank)
comm.Barrier()
if comm.rank == 0: print 'done reading', timer.restart()

if comm.rank == 0: print "making histograms", timer.reset()
if opt.blackhole != None:
  hist, bins = bhlayer.hist()
  plothist(comm, 'bhhist.png', hist)
  save('bhhist-bins.npy', bins)
  save('bhhist-hist.npy', hist)
if opt.star != None:
  hist, bins = starlayer.hist()
  plothist(comm, 'starhist.png', hist)
  save('starhist-bins.npy', bins)
  save('starhist-hist.npy', hist)
if opt.gas != None:
  hist, bins = gaslayer.hist()
  plothist(comm, 'gashist.png', hist)
  save('gashist-bins.npy', bins)
  save('gashist-hist.npy', hist)
  hist, bins = msfrlayer.hist()
  plothist(comm, 'msfrhist.png', hist)
  save('msfrhist-bins.npy', bins)
  save('msfrhist-hist.npy', hist)
comm.Barrier()
if comm.rank == 0: print "done histograms", timer.restart()

