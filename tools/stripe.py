from mpi4py import MPI
from time import clock
from numpy import zeros, fromfile
from numpy import histogram
from numpy import log10
from gadget.plot.image import rasterize
import gadget.plot.render

from numpy import isinf, inf, nan, isnan
from matplotlib.pyplot import plot, clf, title, savefig
from numpy import linspace

def _plothist(layer, filename):
  hist,bins = layer.hist()
  if layer.stripe.comm.rank == 0: 
    clf()
    plot(linspace(0, 1, len(hist)), hist)
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
    self.data.shape = self.stripe.shape
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
  def allocate(self, npoints):
    self.points = zeros(dtype=self.dtype, shape=npoints)
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
  def __init__(self, comm, imagesize):
    self.imagesize = imagesize
    self.pixel_start = imagesize[0] * comm.rank / comm.size
    self.pixel_end = imagesize[0] * (comm.rank + 1)/ comm.size
    self.shape = [self.pixel_end - self.pixel_start, imagesize[1]]
    self.comm = comm
    self.xrange = None
    self.yrange = None
    self.zrange = None
  def set_cut(self, xrange, yrange, zrange):
    self.xrange = [
        xrange[0] + self.pixel_start * (xrange[1] - xrange[0]) / self.imagesize[0],
        xrange[0] + self.pixel_end * (xrange[1] - xrange[0]) / self.imagesize[0] ]
    self.yrange = yrange
    self.zrange = zrange
  def get_cut(self, field):
    if self.xrange == None:
      xrange = [self.pixel_start * field.boxsize[0] / self.imagesize[0], self.pixel_end * field.boxsize[0] /self.imagesize[0]]
      yrange = [0, field.boxsize[1]]
      zrange = [0, field.boxsize[2]]
    else:
      xrange,yrange,zrange = self.xrange, self.yrange, self.zrange
    return xrange, yrange, zrange
  def mkraster(self, field, fieldname, dtype='f4', quick=False):
    xrange,yrange,zrange = self.get_cut(field)
    field['default'] = field[fieldname]
    layer = RasterLayer(self, valuedtype = dtype)
    rasterize(target = layer.pixels, field = field, 
                 xrange = xrange, yrange=yrange, zrange=zrange, quick=quick)
    return layer
  def mkvector(self, field, fieldname, scale):
    field['default'] = field[fieldname]
    xrange,yrange,zrange = self.get_cut(field)
    layer = VectorLayer(self, scale=scale, valuedtype = field['default'].dtype)
    pos = field['locations'].copy()
    pos[:, 0] -= xrange[0]
    pos[:, 1] -= yrange[0]
    pos[:, 0] *= float(self.shape[0]) / (xrange[1] - xrange[0])
    pos[:, 1] *= float(self.shape[1]) / (yrange[1] - yrange[0])
    mask = (pos[:, 0] >= - scale)
    mask &= (pos[:, 0] <= (self.shape[0]+scale))
    mask &= (pos[:, 1] >= (-scale))
    mask &= (pos[:, 1] <= (self.shape[1]+scale))
    layer.allocate(mask.sum())
    layer.points['X'] = pos[mask, 0]
    layer.points['Y'] = pos[mask, 1]
    layer.points['V'] = field['default'][mask]
    return layer 
