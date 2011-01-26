#! python
from optparse import OptionParser, OptionValueError
from gadget.tools import parsearray, parsematrix
parser = OptionParser()
parser.add_option("-N", "--total", type="int", dest="total", help="total number of stripes")
parser.add_option("-p", "--prefix", type="string", default='', help="prefix for input and output filenames")
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
    bins = load(filename)['bins']
    min = bins[0]
    max = bins[-1]
    print filename, "min", min, "max", max
  else: 
    min = None
    max = None
  min = comm.bcast(min)
  max = comm.bcast(max)
  return min, max


if opt.range != None:
  strips = range(opt.range[0], opt.range[1])
else :
  strips = range(0, opt.total)

stripeids_all = [
   strips[rank * len(strips) / comm.size:(rank + 1) * len(strips) / comm.size]
   for rank in range(comm.size)]
steps = max([len(stripeids) for stripeids in stripeids_all])
stripeids = stripeids_all[comm.rank]

if opt.gas != None:
  gasmin, gasmax = loadminmax(comm, opt.prefix+ 'hist-gas.npz')
  msfrmin, msfrmax = loadminmax(comm, opt.prefix+ 'hist-m%s.npz' % opt.gas)
if opt.blackhole != None:
  bhmin, bhmax = loadminmax(comm, opt.prefix + 'hist-bh.npz')
if opt.star != None:
  starmin, starmax = loadminmax(comm, opt.prefix + 'hist-star.npz')

for step in range(steps):
  ses_step = timer.session("step %d" % step)
  ses_reading = timer.session('reading')
  if step < len(stripeids):
    stripeid = stripeids[step]
    stripe = Stripe(rank = stripeid, total = opt.total, imagesize = opt.geometry)
    image = zeros(dtype=('u1', 3), shape = stripe.shape)
  else:
    stripe = None

  if opt.gas != None:
    if stripe != None:
      gaslayer = RasterLayer(stripe, valuedtype='f4')
      gaslayer.fromfile(gaslayer.getfilename(opt.prefix, 'gas', '.rst'))
    ses_reading.checkpoint('gas')
    if stripe != None:
      msfrlayer = RasterLayer(stripe, valuedtype='f4')
      msfrlayer.fromfile(msfrlayer.getfilename(opt.prefix, 'm'+opt.gas, '.rst'))
      msfrlayer.data /= gaslayer.data
    ses_reading.checkpoint('msfr')
  if opt.blackhole != None:
    if stripe != None:
      bhlayer = VectorLayer(stripe, valuedtype='f4', scale=opt.blackhole)
      bhlayer.fromfile(bhlayer.getfilename(opt.prefix, 'bh', '.vec'))
    ses_reading.checkpoint('bh')
  if opt.star != None:
    if stripe != None:
      starlayer = VectorLayer(stripe, valuedtype='f4', scale=opt.star)
      starlayer.fromfile(starlayer.getfilename(opt.prefix, 'bh', '.vec'))
    ses_reading.checkpoint('star')
  ses_reading.end()

  ses_render = timer.session('render')
  if opt.gas != None:
    if stripe != None:
      gaslayer.render(target=image, colormap = gasmap, logscale=True, min=gasmin, max=gasmax)
    ses_render.checkpoint("gas")
    if stripe != None:
      msfrlayer.render(target=image, colormap = msfrmap, logscale=True, min=msfrmin, max=msfrmax)
    ses_render.checkpoint("msfr")

  if opt.star != None:
    if stripe != None:
      starlayer.render(target=image, colormap = starmap, logscale=True, min=starmin, max=starmax)
    ses_render.checkpoint("star")

  if opt.blackhole != None:
    if stripe != None:
      bhlayer.render(target=image, colormap = bhmap, logscale=True, min=bhmin, max=bhmax)
    ses_render.checkpoint("bh")
  ses_render.end()

  ses_write = timer.session("writing")
  if stripe != None:
    if stripe.total < 1000:
      image.tofile(opt.prefix + '%03d.raw' % stripe.rank)
    else:
      image.tofile(opt.prefix + '%04d.raw' % stripe.rank)
  ses_write.end()
  ses_step.end()
