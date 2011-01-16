from mpi4py import MPI
from time import clock
from numpy import zeros, fromfile, array
from numpy import int32
from numpy import histogram
from numpy import log10
from numpy import arange
from numpy import cumsum
from gadget.plot.image import rasterize
import gadget.plot.render
from gadget import Snapshot
from gadget import Field
from numpy import isinf, inf, nan, isnan
from gadget.readers import Readers
from matplotlib.pyplot import plot, clf, title, savefig
from numpy import linspace
from numpy import mean
from numpy import empty
from numpy import append as arrayappend
def _plothist(layer, filename):
  hist,bins = layer.hist()
  if layer.stripe.comm.rank == 0: 
    clf()
    plot(linspace(0, 1, len(hist)), log10(hist))
    title(filename)
    savefig(filename)
class Layer:
  def __init__(self, stripe, valuedtype):
    self.stripe = stripe
    self.data = None
    self.valuedtype = valuedtype
  def min(self, logscale=False):
    if logscale:
      mask = self.data > 0.0
# if logarithm is undefined everywhere, skip this stripe
      if mask.sum() == 0: value = inf
      else: value = self.data[mask].min()
    else: 
      value = self.data.min()
    value = self.stripe.comm.allreduce(value, op = MPI.MIN)
    if logscale:
# if all stripes are skipped, this layer is all below zero
      if isinf(value): value = nan
    return value
  def max(self, logscale=False):
    if logscale:
      mask = self.data > 0.0
# if logarithm is undefined everywhere, skip this stripe
      if mask.sum() == 0: value = -inf
      else: value = self.data[mask].max()
    else: 
      value = self.data.max()
    value = self.stripe.comm.allreduce(value, op = MPI.MAX)
    if logscale:
# if all stripes are skipped, this layer is all below zero
      if isinf(value): value = nan
    return value
  def sum(self):
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
    if max <= min: max = min + 0.1
    if isnan(min) or isnan(max): min,max = 0,1
    if logscale:
      h, bins = histogram(log10(self.data), range=(min, max), bins = bins)
    else:
      h, bins = histogram(self.data, range=(min, max), bins = bins)
    h = self.stripe.comm.allreduce(h, op = MPI.SUM)
    return h, bins
  def plothist(self, filename):
    _plothist(self, filename)
  
class RasterLayer(Layer):
  def __init__(self, stripe, valuedtype, fromfile=None):
    Layer.__init__(self, stripe, valuedtype)
    self.stripe = stripe
    self.pixels = zeros(dtype = self.valuedtype, shape = self.stripe.shape)
    self.data = self.pixels.ravel()
    if fromfile!= None: self.fromfile(fromfile)
  def tofile(self, filename):
    self.data.tofile(filename)
  def fromfile(self, filename):
    self.data = fromfile(filename, dtype=self.valuedtype)
    self.pixels = self.data.view()
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
  
class VectorLayer(Layer):
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
  def __init__(self, comm, imagesize, boxsize, xcut = None, ycut = None, zcut = None):
    self.imagesize = imagesize
    self.pixel_start_all = imagesize[0] * arange(comm.size) / comm.size
    self.pixel_end_all = imagesize[0] * (arange(comm.size) + 1)/ comm.size
    self.pixel_start = self.pixel_start_all[comm.rank]
    self.pixel_end = self.pixel_end_all[comm.rank]
    self.shape = [self.pixel_end - self.pixel_start, imagesize[1]]
    self.comm = comm
    self.boxsize = boxsize
    if xcut != None: self.xcut = xcut
    else: self.xcut = [0, self.boxsize[0]]
    if ycut != None: self.ycut = ycut
    else: self.ycut = [0, self.boxsize[1]]
    if zcut != None: self.zcut = zcut
    else: self.zcut = [0, self.boxsize[2]]
    self.x_start_all = self.pixel_start_all * (self.xcut[1] - self.xcut[0]) / self.imagesize[0] + self.xcut[0]
    self.x_end_all = self.pixel_end_all * (self.xcut[1] - self.xcut[0]) / self.imagesize[0] + self.xcut[0]
    self.x_start = self.x_start_all[comm.rank]
    self.x_end = self.x_end_all[comm.rank]
  def mkraster(self, field, fieldname, dtype='f4', layer=None, quick=False):
    field['default'] = field[fieldname]
    if layer == None:
      layer = RasterLayer(self, valuedtype = dtype)
    rasterize(target = layer.pixels, field = field, 
                 xrange = [self.x_start, self.x_end], yrange=self.ycut, zrange=self.zcut, quick=quick)
    return layer
  def mkvector(self, field, fieldname, scale, layer=None):
    field['default'] = field[fieldname]
    if layer ==None:
      layer = VectorLayer(self, scale=scale, valuedtype = field['default'].dtype)
    pos = field['locations'].copy()
    pos[:, 0] -= self.x_start
    pos[:, 1] -= self.ycut[0]
    pos[:, 0] *= float(self.imagesize[0]) / (self.xcut[1] - self.xcut[0])
    pos[:, 1] *= float(self.imagesize[1]) / (self.ycut[1] - self.ycut[0])
    mask = (pos[:, 0] >= - scale)
    mask &= (pos[:, 0] <= (self.shape[0]+scale))
    mask &= (pos[:, 1] >= (-scale))
    mask &= (pos[:, 1] <= (self.shape[1]+scale))
    layer.append(pos[mask,0], pos[mask,1], field['default'][mask])
    return layer 

  def rebalance(self, field0, values, bleeding = None):
    comm = self.comm
    senddispl = zeros(comm.size, dtype='i4')
    sendcount = zeros(comm.size, dtype='i4')
    recvdispl = zeros(comm.size, dtype='i4')
    recvcount = zeros(comm.size, dtype='i4')
    recvtotal = zeros(comm.size, dtype='i4')
    if bleeding == None:
      bleeding = field0['sml'].max()
    else:
      bleeding = bleeding / self.imagesize[0] * (self.xcut[1] - self.xcut[0])
    if field0 != None:
      sortindex = field0['locations'][:,0].argsort()
      for value in field0:
        field0[value] = field0[value][sortindex]
      senddispl = int32(array(field0['locations'][:,0].searchsorted(self.x_start_all - bleeding, 'left')))
      sendcount = int32(array(field0['locations'][:,0].searchsorted(self.x_end_all + bleeding, 'right')))
      sendcount = sendcount - senddispl

    recvcount = array(comm.alltoall(sendcount))
    recvdispl = int32(array(cumsum([0] + list(recvcount[:-1]))))
    recvtotal_all = comm.allreduce(sendcount, MPI.SUM)
# locations
    sendbuf = None
    recvbuf = zeros(dtype = ('f4', 3), shape = recvtotal_all[comm.rank])
    if field0 != None:
      sendbuf = field0['locations']

      comm.Alltoallv([sendbuf, [3 * sendcount, 3 * senddispl]], 
                    [recvbuf, [3 * recvcount, 3 * recvdispl]])
#values
    field = Field(locations = recvbuf, origin = zeros(3), boxsize = self.boxsize)
    for valuename in values:
      sendbuf = None
      if field0 != None:
        sendbuf = field0[valuename]
      dtype = Readers['d4'][valuename]['dtype']
      if len(dtype.shape) > 0: itemcount = dtype.shape[1]
      else : itemcount = 1
      recvbuf = zeros(dtype = dtype, shape = recvtotal_all[comm.rank])
      comm.Alltoallv([sendbuf, [itemcount * sendcount, itemcount * senddispl]], [recvbuf, [itemcount * recvcount, itemcount * recvdispl]])
      field[valuename] = recvbuf
    return field
