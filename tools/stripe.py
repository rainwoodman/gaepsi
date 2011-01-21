from mpi4py import MPI
from time import clock
from numpy import zeros, fromfile, array
from numpy import int32
from numpy import histogram
from numpy import logspace
from numpy import log10
from numpy import arange
from numpy import cumsum
from gadget.plot.image import rasterize
import gadget.plot.render
from gadget import Snapshot
from gadget import Field
from numpy import isinf, inf, nan, isnan
from gadget.readers import Readers
from numpy import linspace
from numpy import mean
from numpy import empty
from numpy import append as arrayappend
from gadget import ccode
class Layer:
  def __init__(self, stripe, valuedtype):
    self.stripe = stripe
    self.data = None
    self.valuedtype = valuedtype
  def min(self, logscale=False):
    if len(self.data) == 0: return nan
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
    if len(self.data) == 0: return nan
    value = self.data.max()
    if logscale and value <= 0.0:
      value = nan
    value = self.stripe.comm.allreduce(value, op = MPI.MAX)
    return value
  def sum(self):
    if len(self.data) == 0: return 0.0
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
  def tofile(self, filename):
    self.data.tofile(filename)
  def fromfile(self, filename):
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
  def __init__(self, imagesize, rank=None, total=None, comm = None, boxsize = None, xcut = None, ycut = None, zcut = None):
    """if comm is given, the stripes are constructed with rank=comm.rank, and total=comm.size
       comm is not required if Layer.{hist, max, min} and rebalance are not called.
       boxsize is not required if the stripe is not used to make layers(ie mkvector/mkraster are not invoked,
               use the 3D boxsize that will be projected to this image
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
      
    if boxsize != None:
      self.boxsize = boxsize
      if xcut != None: self.xcut = xcut
      else: self.xcut = [0, self.boxsize[0]]
      if ycut != None: self.ycut = ycut
      else: self.ycut = [0, self.boxsize[1]]
      if zcut != None: self.zcut = zcut
      else: self.zcut = [0, self.boxsize[2]]
      self.x_start_all = self.pixel_start_all * (self.xcut[1] - self.xcut[0]) / self.imagesize[0] + self.xcut[0]
      self.x_end_all = self.pixel_end_all * (self.xcut[1] - self.xcut[0]) / self.imagesize[0] + self.xcut[0]
      self.x_start = self.x_start_all[rank]
      self.x_end = self.x_end_all[rank]

  def mkraster(self, field, fieldname, dtype='f4', layer=None, quick=False):
    if type(fieldname) != list:
      fieldname = [fieldname]
    
    if layer == None:
      layer = [RasterLayer(self, valuedtype = dtype) for f in fieldname]
    if type(layer) != list:
      layer = [layer]
    targets = [l.pixels for l in layer]
    rasterize(targets = targets, field = field, values = fieldname,
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
    if field0 != None and field0.numpoints != 0:
      if bleeding == None:
        bleeding = field0['sml'].max()
      else:
        bleeding = bleeding * (self.xcut[1] - self.xcut[0]) / self.imagesize[0]
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
    recvbuf = zeros(dtype = ('f4', 3), shape = recvtotal_all[comm.rank])
    if field0 != None and field0.numpoints != 0:
      sendbuf = field0['locations']
    else:
      sendbuf = zeros(dtype= ('f4', 3), shape = 1)

    comm.Alltoallv([sendbuf, [3 * sendcount, 3 * senddispl]], 
                    [recvbuf, [3 * recvcount, 3 * recvdispl]])
#values
    field = Field(locations = recvbuf, origin = zeros(3), boxsize = self.boxsize)
    for valuename in values:
      dtype = Readers['d4'][valuename]['dtype']
      recvbuf = zeros(dtype = dtype, shape = recvtotal_all[comm.rank])
      if field0 != None and field0.numpoints != 0:
        sendbuf = field0[valuename]
      else:
        sendbuf = zeros(dtype = dtype, shape = 1)
      if len(dtype.shape) > 0: itemcount = dtype.shape[1]
      else : itemcount = 1
      
      comm.Alltoallv([sendbuf, [itemcount * sendcount, itemcount * senddispl]], 
                     [recvbuf, [itemcount * recvcount, itemcount * recvdispl]])
      field[valuename] = recvbuf
    return field
