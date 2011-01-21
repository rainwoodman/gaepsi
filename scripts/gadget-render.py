#! python
from optparse import OptionParser, OptionValueError
from gadget.tools import parsearray, parsematrix
parser = OptionParser()
parser.add_option("-N", "--total", type="int", dest="total", help="total number of stripes")
parser.add_option("-r", "--range", dest="range", type="string", help="stripe range to process",
     action="callback", callback=parsearray, callback_kwargs=dict(sep=',', dtype='i4', len=2))
parser.add_option("-g", "--geometry", dest="geometry",  type="string",
     action="callback", callback=parsearray, callback_kwargs=dict(sep='x', dtype='i4', len=2))
parser.add_option("-B", "--blackhole", dest="blackhole", type="float", help="process blackhole with the give circle size")
parser.add_option("-G", "--gas", dest="gas", type="string", help="process gas produce the given mass weighted field and mass field")
parser.add_option("-S", "--star", dest="star", type="float", help="process star with the given circle size")

opt, args = parser.parse_args()
if opt.geometry == None:
  parser.error("specify --geometry")

from gadget.plot.render import Colormap

bhmap = Colormap(levels = [0, 0.5, 1.0],
                    r = [0.0, 0.0, 0.0],
                    g = [1.0, 1.0, 1.0],
                    b = [0.0, 0.0, 0.0],
                    v = [0.2, 0.5, 1.0])
starmap = Colormap(levels = [0, 0.5, 1.0],
                    r = [1.0, 1.0, 1.0],
                    g = [1.0, 1.0, 1.0],
                    b = [1.0, 1.0, 1.0],
                    v = [1.0, 1.0, 1.0],
                    a = [0.7, 0.7, 0.7])

gasmap = Colormap(levels =[0, 0.05, 0.2, 0.5, 0.6, 0.8, 1.0],
                      r = [0, 0.1 ,0.5, 1.0, 0.2, 0.0, 0.0],
                      g = [0, 0  , 0.2, 1.0, 0.2, 0.0, 0.0],
                      b = [0, 0  , 0  , 0.0, 0.4, 0.8, 1.0])

msfrmap = Colormap(levels =[0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0],
                      h = [257, 257  , 257, 257, 257, 257, 257],
                      s = [1, 1  , 1.0, 1.0, 1.0, 1.0, 1.0],
                      v = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
                      a = [0 , 0,  0.4, 0.5, 0.6, 0.8, 0.8])

from mpi4py import MPI
comm = MPI.COMM_WORLD
if comm.rank == 0: print opt
from gadget.tools import timer
from gadget.tools.stripe import *
from numpy import zeros
from numpy import load
def loadminmax(comm, filename):
  if comm.rank == 0:
     bins = load(filename)
     print filename, "min", bins[0], "max", bins[-1]
  else: bins = None
  bins = comm.bcast(bins)
  return bins[0], bins[-1]


strips = range(opt.range[0], opt.range[1])
stripeids_all = [
   strips[rank * len(strips) / comm.size:(rank + 1) * len(strips) / comm.size]
   for rank in range(comm.size)]
steps = max([len(stripeids) for stripeids in stripeids_all])
stripeids = stripeids_all[comm.rank]

if opt.gas != None:
  gasmin, gasmax = loadminmax(comm, 'gashist-bins.npy')
  msfrmin, msfrmax = loadminmax(comm, 'msfrhist-bins.npy')
if opt.blackhole != None:
  bhmin, bhmax = loadminmax(comm, 'bhhist-bins.npy')
if opt.star != None:
  starmin, starmax = loadminmax(comm, 'starhist-bins.npy')

for step in range(steps):
  comm.Barrier()
  if comm.rank == 0: print 'step ', step, timer.restart()
  if step < len(stripeids):
    stripeid = stripeids[step]
    stripe = Stripe(rank = stripeid, total = opt.total, imagesize = opt.geometry)
    if opt.gas != None:
      gaslayer = RasterLayer(stripe, valuedtype='f4')
      msfrlayer = RasterLayer(stripe, valuedtype='f4')
      gaslayer.fromfile('%03d-gas.rst' % stripeid)
      msfrlayer.fromfile('%03d-msfr.rst' % stripeid)
      msfrlayer.data /= gaslayer.data
    if opt.blackhole != None:
      bhlayer = VectorLayer(stripe, valuedtype='f4', scale=opt.blackhole)
      bhlayer.fromfile('%03d-bh.vec' % stripeid)
    if opt.star != None:
      starlayer = VectorLayer(stripe, valuedtype='f4', scale=opt.star)
      starlayer.fromfile('%03d-star.vec' % stripeid)
  else:
    stripe = None
  comm.Barrier()
  if comm.rank == 0: print "reading done", timer.reset()

  if stripe != None:
    image = zeros(dtype=('u1', 3), shape = stripe.shape)
    if opt.gas != None:
      gaslayer.render(target=image, colormap = gasmap, logscale=True, min=gasmin, max=gasmax)
      msfrlayer.render(target=image, colormap = msfrmap, logscale=True, min=msfrmin, max=msfrmax)

  comm.Barrier()
  if comm.rank == 0: print "rendering raster done", timer.reset()
  if stripe != None:
    if opt.star != None:
      starlayer.render(target=image, colormap = starmap, logscale=True, min=starmin, max=starmax)
    if opt.blackhole != None:
      bhlayer.render(target=image, colormap = bhmap, logscale=True, min=bhmin, max=bhmax)
  comm.Barrier()
  if comm.rank == 0: print "rendering vector done", timer.reset()

  if stripe != None:
    image.tofile('%03d.raw' % stripeid)
  comm.Barrier()
  if comm.rank == 0: print 'images wrote', timer.restart()
