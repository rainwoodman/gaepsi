from numpy import fromfile
from numpy import zeros, dtype, ones
from numpy import uint8, array, linspace
from numpy import int32, float32, int16
from numpy import log10
from gaepsi import ccode

def icmap(levels, cmap, bins):
  cmap = array(cmap)
  detail = zeros(bins)
  for i in range(len(cmap) -1):
    start = int(levels[i] * (bins - 1))
    end = int(levels[i + 1] * (bins -1))
    detail[start:end] = linspace(cmap[i], cmap[i+1], end - start, endpoint=False)
  detail[-1] = cmap[-1]
  return detail

class Valuemap:
  def __init__(self, levels, bins = 1024 * 16, **kwargs):
    self.levels = levels
    self.bins = bins
    self.values = {}
    self.table = {}
    for key in kwargs:
      if not len(kwargs[key]) == len(levels): raise ValueError("length of a component %s differ from levels %d" % (key, len(levels)))
      self.add(key, kwargs[key])
  def add(self, component, values):
    self.values[component] = values
    self.table[component] = icmap(self.levels, values, self.bins)
class Colormap(Valuemap):
  def __init__(self, levels, bins=1024*16, **kwargs):
    Valuemap.__init__(self, levels, bins, **kwargs)
    if 'h' in kwargs:
      self.hsv2rgb()
    if not 'a' in kwargs:
      self.add('a', ones(len(self.levels), dtype='f4'))

  def hsv2rgb(self):
    hi = self.table['h']
    si = self.table['s']
    vi = self.table['v']
    f = hi % 60
    l = uint8(hi / 60)
    l = l % 6
    p = vi * (1 - si)
    q = vi * (1 - si * f / 60)
    t = vi * (1 - si * (60 - f) / 60)
    ri = zeros(len(vi))
    gi = zeros(len(vi))
    bi = zeros(len(vi))

    ri[l==0] = vi[l==0]
    gi[l==0] = t[l==0]
    bi[l==0] = p[l==0]

    ri[l==1] = q[l==1]
    gi[l==1] = vi[l==1]
    bi[l==1] = p[l==1]

    ri[l==2] = p[l==2]
    gi[l==2] = vi[l==2]
    bi[l==2] = t[l==2]
  
    ri[l==3] = p[l==3]
    gi[l==3] = q[l==3]
    bi[l==3] = vi[l==3]

    ri[l==4] = t[l==4]
    gi[l==4] = p[l==4]
    bi[l==4] = vi[l==4]

    ri[l==5] = vi[l==5]
    gi[l==5] = p[l==5]
    bi[l==5] = q[l==5]
    self.table['r'] = ri
    self.table['g'] = gi
    self.table['b'] = bi

def circle(target, X, Y, V, scale, min=None, max=None, logscale=False, colormap=None): 
  if colormap == None:
    colormap = Colormap(levels = [0, 0.2, 0.4, 0.6, 1.0], 
                         r=[0, 0.5, 1.0, 1.0, 0.2], 
                         g=[0, 0.0, 0.5, 1.0, 0.2], 
                         b=[0.0, 0.0, 0.0, 0.3, 1.0])
  if len(target.shape) != 3:
     raise ValueError("has to be a rgb bitmap! expecting shape=(#,#,3)")
  image = target
  if min == None: min = V.min()
  if max == None: max = V.max()
  if min == max: 
    min = min - 0.5
    max = max + 0.5

  ccode.circle(target = target, X = X, Y = Y, V = V,
    min = min, max = max,
    scale = scale,
    logscale=logscale,
    cmapr=colormap.table['r'],
    cmapg=colormap.table['g'],
    cmapb=colormap.table['b'],
    cmapa=colormap.table['a'],
    cmapv=colormap.table['v'])

def line (target, X, Y, VX,VY, scale, min=None, max=None, logscale=False, colormap=None): 
  if colormap == None:
    colormap = Colormap(levels = [0, 0.2, 0.4, 0.6, 1.0], 
                         r=[0, 0.5, 1.0, 1.0, 0.2], 
                         g=[0, 0.0, 0.5, 1.0, 0.2], 
                         b=[0.0, 0.0, 0.0, 0.3, 1.0])
  if len(target.shape) != 3:
     raise ValueError("has to be a rgb bitmap! expecting shape=(#,#,3)")
  image = target
#fix me: use the norm! how to avoid creating a new array?
  if min == None: min = array([VX.min(), VY.min()]).max()
  if max == None: max = array([VX.max(), VY.max()]).max()
  if min == max: 
    min = min - 0.5
    max = max + 0.5

  ccode.line(target = target, X = X, Y = Y, VX = VX, VY=VY,
    min = min, max = max,
    scale = scale,
    logscale=logscale,
    cmapr=colormap.table['r'],
    cmapg=colormap.table['g'],
    cmapb=colormap.table['b'],
    cmapa=colormap.table['a'],
    cmapv=colormap.table['v'])

def color(target, raster, min, max, logscale=True, colormap=None):
  if colormap == None:
    colormap = Colormap(levels = [0, 0.2, 0.4, 0.6, 1.0], 
                         r=[0, 0.5, 1.0, 1.0, 0.2], 
                         g=[0, 0.0, 0.5, 1.0, 0.2], 
                         b=[0.0, 0.0, 0.0, 0.3, 1.0])

  ccode.color(target=target, raster = raster, min = min, max = max, 
    logscale = logscale, 
    cmapr=colormap.table['r'],
    cmapg=colormap.table['g'],
    cmapb=colormap.table['b'],
    cmapa=colormap.table['a'])
  return target

