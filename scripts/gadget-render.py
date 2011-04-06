#! python
from gadget.tools import timer
from optparse import OptionParser, OptionValueError
from gadget.tools.cmdline import parsearray, parsematrix
parser = OptionParser(conflict_handler="resolve")
parser.add_option("-N", "--total", type="int", dest="total", help="total number of stripes")
parser.add_option("-p", "--prefix", type="string", default='', help="prefix for input and output filenames")
parser.add_option("-r", "--range", dest="range", type="string", help="stripe range to process",
     action="callback", callback=parsearray, callback_kwargs=dict(sep=',', dtype='i4', len=2))
parser.add_option("-g", "--geometry", dest="geometry",  type="string",
     action="callback", callback=parsearray, callback_kwargs=dict(sep='x', dtype='i4', len=2))
parser.add_option("-B", "--blackhole", dest="blackhole", type="float", help="draw blackholes with the give circle size")
parser.add_option("-G", "--gas", dest="gas", action="store_true", help="render gas mass field")
parser.add_option('-S', "--sfr", dest="sfr", action="store_true", help="render gas star formation rate field")
parser.add_option('-T', "--temp", dest="temp", action="store_true", help="render gas temperature field")
parser.add_option("-S", "--star", dest="star", type="float", help="draw stars with the given circle size")
parser.add_option("-o", "--output", dest="output", type="string", help="put output to one big file with MPIIO")
parser.add_option("-z", "--zip", dest="zip", action="store_true", help="unzip the input instead assuming a direct dump")
parser.add_option("-c", "--colormap", dest="colormap", help="use this file for bhmap, starmap ans gasmap, tempmap")

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
mtempmap = Colormap(levels =[0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0],
                      h = [180, 160, 140, 120, 100, 90, 90],
                      s = [1, 1  , 1.0, 1.0, 1.0, 1.0, 1.0],
                      v = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
                      a = [0 , 0,  0.4, 0.5, 0.6, 0.8, 0.8])
msfrmap = Colormap(levels =[0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0],
                      h = [257, 257  , 257, 257, 257, 257, 257],
                      s = [1, 1  , 1.0, 1.0, 1.0, 1.0, 1.0],
                      v = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
                      a = [0 , 0,  0.4, 0.5, 0.6, 0.8, 0.8])

if opt.colormap != None:
  execfile(opt.colormap)

from gadget.tools import _MPI as MPI
comm = MPI.COMM_WORLD
if comm.rank == 0: print opt

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
if opt.sfr != None:
  msfrmin, msfrmax = loadminmax(comm, opt.prefix+ 'hist-msfr.npz')
if opt.temp!= None:
  mtempmin, mtempmax = loadminmax(comm, opt.prefix+ 'hist-mtemp.npz')
if opt.blackhole != None:
  bhmin, bhmax = loadminmax(comm, opt.prefix + 'hist-bh.npz')
if opt.star != None:
  starmin, starmax = loadminmax(comm, opt.prefix + 'hist-star.npz')
if opt.output != None:
  mpifile = MPI.File.Open(comm, opt.output, MPI.MODE_WRONLY + MPI.MODE_CREATE)
  mpifile.Set_view(disp = 0, etype=MPI.BYTE, filetype=MPI.BYTE)
else:
  mpifile = None

for step in range(steps):
  ses_step = timer.session("step %d" % step)
  ses_reading = timer.session('reading')
  if step < len(stripeids):
    stripeid = stripeids[step]
    stripe = Stripe(rank = stripeid, total = opt.total, imagesize = opt.geometry)
    if stripe.total < 1000:
      fn = opt.prefix + '%03d.raw' % stripe.rank 
    else:
      fn = opt.prefix + '%04d.raw' % stripe.rank
    if mpifile == None and os.path.exists(fn):
      ses_step.end()
      continue
    image = zeros(dtype=('u1', 3), shape = stripe.shape)
  else:
    stripe = None

  if opt.gas != None or opt.sfr !=None or opt.temp != None:
    if stripe != None:
      gaslayer = RasterLayer(stripe, valuedtype='f4')
      gaslayer.fromfile(gaslayer.getfilename(opt.prefix, 'gas', '.rst'), zip=opt.zip)
    ses_reading.checkpoint('gas')
  if opt.sfr != None:
    if stripe != None:
      msfrlayer = RasterLayer(stripe, valuedtype='f4')
      msfrlayer.fromfile(msfrlayer.getfilename(opt.prefix, 'msfr', '.rst'), zip=opt.zip)
      msfrlayer.data /= gaslayer.data
    ses_reading.checkpoint('msfr')
  if opt.temp != None:
    if stripe != None:
      mtemplayer = RasterLayer(stripe, valuedtype='f4')
      mtemplayer.fromfile(mtemplayer.getfilename(opt.prefix, 'mtemp', '.rst'), zip=opt.zip)
      mtemplayer.data /= gaslayer.data
    ses_reading.checkpoint('mtemp')
  if opt.blackhole != None:
    if stripe != None:
      bhlayer = ScatterLayer(stripe, valuedtype='f4', scale=opt.blackhole)
      bhlayer.fromfile(bhlayer.getfilename(opt.prefix, 'bh', '.vec'))
    ses_reading.checkpoint('bh')
  if opt.star != None:
    if stripe != None:
      starlayer = ScatterLayer(stripe, valuedtype='f4', scale=opt.star)
      starlayer.fromfile(starlayer.getfilename(opt.prefix, 'star', '.vec'))
    ses_reading.checkpoint('star')
  ses_reading.end()

  ses_render = timer.session('render')
  if opt.gas != None:
    if stripe != None:
      gaslayer.render(target=image, colormap = gasmap, logscale=True, min=gasmin, max=gasmax)
      del gaslayer
    ses_render.checkpoint("gas")
  if opt.sfr != None:
    if stripe != None:
      msfrlayer.render(target=image, colormap = msfrmap, logscale=True, min=msfrmin, max=msfrmax)
      del msfrlayer
    ses_render.checkpoint("sfr")
  if opt.temp!= None:
    if stripe != None:
      mtemplayer.render(target=image, colormap = mtempmap, logscale=True, min=mtempmin, max=mtempmax)
      del mtemplayer
    ses_render.checkpoint("msfr")

  if opt.star != None:
    if stripe != None:
      starlayer.render(target=image, colormap = starmap, logscale=True, min=starmin, max=starmax)
      del starlayer
    ses_render.checkpoint("star")

  if opt.blackhole != None:
    if stripe != None:
      bhlayer.render(target=image, colormap = bhmap, logscale=True, min=bhmin, max=bhmax)
      del bhlayer
    ses_render.checkpoint("bh")
  ses_render.end()

  ses_write = timer.session("writing")
  if stripe != None:
    flat = image.ravel()
    buflen=8192*1024
    if mpifile != None:
      for i in range(0, flat.size, buflen):
        data = flat[i:i+buflen]
        mpifile.Write_at((stripe.pixel_start * stripe.shape[1]) * 3 + i, [data.data, MPI.BYTE])
    else :
      if stripe.total < 1000:
        datafile = file(opt.prefix + '%03d.raw' % stripe.rank, 'w')
      else:
        datafile = file(opt.prefix + '%04d.raw' % stripe.rank, 'w')
      for i in range(0, flat.size, buflen):
        data = flat[i:i+buflen]
        data.tofile(datafile)
      del datafile
    del image
    del flat
  ses_write.end()
  ses_step.end()
if mpifile != None: mpifile.Close()
