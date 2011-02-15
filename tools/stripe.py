from gadget.tools import _MPI as MPI
from numpy import zeros, fromfile, array
from numpy import int32
from numpy import histogram
from numpy import logspace
from numpy import log10
from numpy import arange
from numpy import cumsum
from gadget.plot.image import rasterize
import gadget.plot.render
from gadget.field import Field
from numpy import isinf, inf, nan, isnan
from gadget.readers import Readers
from numpy import linspace
from numpy import mean
from numpy import empty
from numpy import append as arrayappend
from numpy import fmax
from gadget import ccode
from gadget.tools.zip import fromzipfile
from gadget.tools.zip import tozipfile
class Layer:
  def __init__(self, stripe, valuedtype):
    self.stripe = stripe
    self.data = None
    self.valuedtype = valuedtype
    self.numparticles = 0

  def getfilename(self, prefix, signature, postfix):
    if self.stripe.total < 1000:
      fmt = '%03d-%s'
    else:
      fmt = '%04d-%s'
    return prefix + fmt %(self.stripe.rank, signature) + postfix

  def min(self, logscale=False):
    if len(self.data) == 0: 
      value = inf
    else:
      if logscale:
        value = ccode.pmin.reduce(self.data)
      else: 
        value = self.data.min()
    value = self.stripe.comm.allreduce(value, op = MPI.MIN)
    if logscale:
# if all stripes are skipped, this layer is all below zero
      if isinf(value): value = nan
    return value
  def max(self, logscale=False):
    if len(self.data) == 0: 
      value = -inf
    else:
      value = fmax.reduce(self.data)
      if logscale and value <= 0.0:
        value = -inf
    value = self.stripe.comm.allreduce(value, op = MPI.MAX)
    if logscale and value == -inf:
      value = nan
    return value
  def sum(self):
    if len(self.data) == 0: 
      value = 0.0
    else:
      value = self.data.sum()
    value = self.stripe.comm.allreduce(value, op = MPI.SUM)
    return value
  def hist(self, bins=100, min = None, max = None, logscale=True):
    if logscale:
      if min == None: min = log10(self.min(logscale=True))
      if max == None: max = log10(self.max(logscale=True))
    else:
      if min == None: min = self.min()
      if max == None: max = self.max()
    if max == min: 
      max = min + 0.1
      min = min - 0.1
    if isnan(min) or isnan(max): min,max = 0,1
    if logscale:
      edges = logspace(min, max, bins + 1)
      h, bins = histogram(self.data, bins = edges)
      bins = log10(bins)
    else:
      h, bins = histogram(self.data, range=(min, max), bins = bins)
    h = self.stripe.comm.allreduce(h, op = MPI.SUM)
    return h, bins
  
class RasterLayer(Layer):
  def __init__(self, stripe, valuedtype, fromfile=None):
    Layer.__init__(self, stripe, valuedtype)
    self.stripe = stripe
    self.pixels = zeros(dtype = self.valuedtype, shape = self.stripe.shape)
    self.data = self.pixels.ravel()
    if fromfile!= None: self.fromfile(fromfile)
  def tofile(self, filename, zip=False):
    if zip:
      tozipfile(filename, self.data)
    else:
      self.data.tofile(filename)
  def fromfile(self, filename, zip=False):
    if zip:
      fromzipfile(filename, self.data)
    else:
      del self.data
      del self.pixels
      self.data = fromfile(filename, dtype=self.valuedtype)
      self.pixels = self.data.view()
      if self.stripe.shape[0] * self.stripe.shape[1] != self.pixels.shape:
        print "stripe mismatch", self.pixels.shape, self.stripe.shape, self.stripe.rank, self.stripe.total
      self.pixels.shape = self.stripe.shape

  def render(self, target, colormap, min=None, max=None, logscale=True):
    if logscale:
      if min == None: min = log10(self.min(logscale=True))
      if max == None: max = log10(self.max(logscale=True))
    else:
      if min == None: min = self.min()
      if max == None: max = self.max()
    if max <= min: max = min + 0.1
    if isnan(min) or isnan(max): min,max = 0,1
    gadget.plot.render.color(target = target, raster = self.pixels, logscale = logscale, min = min, max = max, colormap = colormap)
  
class ScatterLayer(Layer):
  def __init__(self, stripe, valuedtype, scale, fromfile=None):
    Layer.__init__(self, stripe, valuedtype)
    self.stripe = stripe
    self.scale = scale
    self.dtype = [('X', 'f4'), ('Y', 'f4'), ('V', self.valuedtype)]
    self.points = None
    if fromfile != None: self.fromfile(fromfile)
  def append(self, X,Y,V):
    points = empty(dtype=self.dtype, shape = len(X))
    points['X'] = X
    points['Y'] = Y
    points['V'] = V
    if self.points!=None:
      self.points = arrayappend(self.points, points)
    else :
      self.points = points
    self.data = self.points['V']
  def tofile(self, filename):
    self.points.tofile(filename)
  def fromfile(self, filename):
    self.points = fromfile(filename, dtype=self.dtype)
    self.data = self.points['V']
  def render(self, target, colormap, min=None, max=None, logscale=True):
    if logscale:
      if min == None: min = log10(self.min(logscale=True))
      if max == None: max = log10(self.max(logscale=True))
    else:
      if min == None: min = self.min()
      if max == None: max = self.max()
    if max <= min: max = min + 0.1
    if isnan(min) or isnan(max): min,max = 0,1
    gadget.plot.render.circle(target = target, X=self.points['X'], Y=self.points['Y'], V=self.points['V'], 
         scale = self.scale, min = min, max = max, colormap = colormap, logscale = logscale)

class Stripe:
  def __init__(self, imagesize, rank=None, total=None, comm = None, xcut = None, ycut = None, zcut = None):
    """if comm is given, the stripes are constructed with rank=comm.rank, and total=comm.size
       comm is not required if Layer.{hist, max, min} and rebalance are not called.
       {xyz}cut defaults to the entire space"""
    self.comm = comm
    if comm != None:
      rank = comm.rank
      total = comm.size

    self.imagesize = imagesize
    self.pixel_start_all = imagesize[0] * arange(total) / total
    self.pixel_end_all = imagesize[0] * (arange(total) + 1)/ total
    self.pixel_start = self.pixel_start_all[rank]
    self.pixel_end = self.pixel_end_all[rank]
    self.shape = [self.pixel_end - self.pixel_start, imagesize[1]]
    self.rank = rank
    self.total = total
    self.xcut = xcut
    self.ycut = ycut
    self.zcut = zcut

  def get_cuts(self, boxsize):
    class __CUT(object):
      pass
    rt = __CUT()
    if self.xcut != None: rt.xcut = self.xcut
    else: rt.xcut = [0, boxsize[0]]
    if self.ycut != None: rt.ycut = self.ycut
    else: rt.ycut = [0, boxsize[1]]
    if self.zcut != None: rt.zcut = self.zcut
    else: rt.zcut = [0, boxsize[2]]
    rt.xres = (rt.xcut[1] - rt.xcut[0]) / self.imagesize[0]
    rt.yres = (rt.ycut[1] - rt.ycut[0]) / self.imagesize[1]
    rt.x_start_all = self.pixel_start_all * rt.xres + rt.xcut[0]
    rt.x_end_all = self.pixel_end_all * rt.xres + rt.xcut[0]
    rt.x_start = rt.x_start_all[self.rank]
    rt.x_end = rt.x_end_all[self.rank]
    return rt

  def mkraster(self, field, fieldname, layer=None, dtype='f4', quick=False):
    if type(fieldname) != list:
      fieldname = [fieldname]
    if layer == None:
      layer = [RasterLayer(self, valuedtype = dtype) for f in fieldname]
    if type(layer) != list:
      layer = [layer]
    targets = [l.pixels for l in layer]
    cuts = self.get_cuts(field.boxsize)
    numparticles = rasterize(targets = targets, field = field, values = fieldname,
                 xrange = [cuts.x_start, cuts.x_end], yrange=cuts.ycut, zrange=cuts.zcut, quick=quick)
    for l in layer:
      l.numparticles += numparticles

    return layer
  def mkscatter(self, field, fieldname, scale, layer=None):
    if layer ==None:
      layer = ScatterLayer(self, scale=scale, valuedtype = field[fieldname].dtype)
    pos = field['locations'].copy()
    cuts = self.get_cuts(field.boxsize)
    pos[:, 0] -= cuts.x_start
    pos[:, 1] -= cuts.ycut[0]
    pos[:, 0] /= cuts.xres
    pos[:, 1] /= cuts.yres

    mask = (pos[:, 0] >= - scale)
    mask &= (pos[:, 0] <= (self.shape[0]+scale))
    mask &= (pos[:, 1] >= (-scale))
    mask &= (pos[:, 1] <= (self.shape[1]+scale))
    layer.append(pos[mask,0], pos[mask,1], field[fieldname][mask])
    layer.numparticles += mask.sum()
    return layer 

  def rebalance(self, field, bleeding = None):
    comm = self.comm
    senddispl = zeros(comm.size, dtype='i4')
    sendcount = zeros(comm.size, dtype='i4')
    recvdispl = zeros(comm.size, dtype='i4')
    recvcount = zeros(comm.size, dtype='i4')
    recvtotal = zeros(comm.size, dtype='i4')

    cuts = self.get_cuts(field.boxsize)
    if field.numpoints > 0:
      if bleeding == None:
        bleeding = field['sml'].max() * 1.0
      else:
        bleeding = bleeding * cuts.xres
      sortindex = field['locations'][:,0].argsort()
      for comp in field.dict.keys():
        field[comp] = field[comp][sortindex]

      senddispl = int32(array(field['locations'][:,0].searchsorted(cuts.x_start_all - bleeding, 'left')))
      sendcount = int32(array(field['locations'][:,0].searchsorted(cuts.x_end_all + bleeding, 'right'))) - senddispl

    bleedings = comm.allgather(bleeding)

    recvcount = array(comm.alltoall(sendcount))
    recvdispl = int32(array(cumsum([0] + list(recvcount[:-1]))))
    recvtotal_all = comm.allreduce(sendcount, MPI.SUM)

    field.numpoints = recvtotal_all[comm.rank]
    for comp in field.dict.keys():
      dtype = field[comp].dtype
      shape = list(field[comp].shape)
      shape[0] = recvtotal_all[comm.rank]
      recvbuf = zeros(dtype = dtype, shape = shape)
      sendbuf = field[comp]
      if len(shape) > 1: itemcount = shape[1]
      else : itemcount = 1
      
#      for i in range(comm.size):
#        if comm.rank == i: print i, 'comp', comp, 'itemcount', itemcount
#        if comm.rank == i: print i, sendcount, senddispl, recvcount, recvdispl
#        if comm.rank == i: print i, sendbuf.shape, recvbuf.shape
#        comm.Barrier()

      comm.Alltoallv([sendbuf, [itemcount * sendcount, itemcount * senddispl]], 
                     [recvbuf, [itemcount * recvcount, itemcount * recvdispl]])
#      if comm.rank == 0: print 'comp', comp, 'done'
      field[comp] = recvbuf
    return field
