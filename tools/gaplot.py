#! python
from numpy import asarray
from matplotlib.pyplot import *
from gadget.snapshot import Snapshot
from gadget.field import Field, Cut

from gadget.plot.image import rasterize
from gadget.plot.render import Colormap
from gadget.tools.streamplot import streamplot
from gadget import ccode
from numpy import zeros, linspace, meshgrid, log10, average, vstack,absolute,fmin,fmax
from numpy import isinf, nan, argsort
from numpy import tile, unique, sqrt, nonzero
from numpy import ceil
gasmap = Colormap(levels =[0, 0.05, 0.2, 0.5, 0.6, 0.9, 1.0],
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

def __pycmap(gacmap):
  from matplotlib.colors import ListedColormap
  r = gacmap.table['r']
  g = gacmap.table['g']
  b = gacmap.table['b']
  rt = ListedColormap(vstack((r, g, b)).T)
  rt.set_over((r[-1], g[-1], b[-1]))
  rt.set_under((r[0], g[0], b[0]))
  rt.set_bad(color='k', alpha=0)
  return rt

pygasmap = __pycmap(gasmap)


class Context:
  def __init__(self):
    self.format = 'hydro3200'
    self.shape = (1000,1000)
    self.cache = {}

  def zoom(self, center=None, size=None):
    if center == None:
      center = self.cut.center
    if size == None:
      size = self.cut.size
    self.cut=Cut(center=center, size=size)
    self.cache.clear()

  def select(self, mask):
    self.gas.set_mask(mask)
    self.cache.clear()

  def use(self, snapname, format, components={}):
    self.components = components
    self.components['mass'] = 'f4'
    self.components['sml'] = 'f4'
    self.components['vel'] = ('f4', 3)
    self.snapname = snapname
    self.format = format

  @property
  def redshift(self):
    return self.C['Z']

  @property
  def gas(self):
    return self.__gas

  @property
  def bh(self):
    return self.__bh

  @property
  def star(self):
    return self.__star

  def read(self, fids=None, cut=None):
    if fids != None:
      snapnames = [self.snapname % i for i in fids]
    else:
      snapnames = [self.snapname]

    self.__gas = Field(components=self.components, cut=cut)
    self.__bh = Field(components={'bhmass':'f4'}, cut=cut)
    self.__star = Field(components={'sft':'f4'}, cut=cut)
    for snapname in snapnames:
      snap = Snapshot(snapname, self.format)
      self.gas.add_snapshot(snap, ptype = 0, components=self.components.keys())
      self.bh.add_snapshot(snap, ptype = 5, components=['bhmass'])
      self.star.add_snapshot(snap, ptype = 4, components=['sft'])
      print snapname , 'loaded', 'gas particles', self.gas.numpoints
    self.C = snap.C.copy()
    self.cache.clear()
    if cut == None:
      self.cut = Cut(xcut=[0, snap.C['L']], ycut=[0, snap.C['L']], zcut=[0, snap.C['L']])
    else:
      self.cut = cut

  def rotate(self, *args, **kwargs):
    self.gas.rotate(*args, **kwargs)
    self.bh.rotate(*args, **kwargs)
    self.star.rotate(*args, **kwargs)

    self.cache.clear()

  def bhshow(self, radius=4, count=100):
    from matplotlib.collections import CircleCollection
    mask = self.cut.select(self.bh['locations'])
    X = self.bh['locations'][mask,0]
    Y = self.bh['locations'][mask,1]
    R = log10(self.bh['bhmass'][mask])
    if len(R) == 0: return
    if len(R) > 1:
      R -= R.min()
    R /=R.max()
    ind = (-R).argsort()
    X = X[ind[0:count]]
    Y = Y[ind[0:count]]
    R = R[ind[0:count]]
    R*=radius**2
    R.clip(min=4, max=radius**2)
    scatter(X,Y, s=R*radius, marker='o', edgecolor='green', facecolor=(0,1,0,0.0))
  #  col = CircleCollection(offsets=zip(X.flat,Y.flat), sizes=(R * radius)**2, edgecolor='green', facecolor='none', transOffset=gca().transData)
  #  gca().add_collection(col)
    axis([self.cut['x'][0], self.cut['x'][1], self.cut['y'][0], self.cut['y'][1]])

  def circle(self, *args, **kwargs):
    from matplotlib.patches import Circle
    c = Circle(*args, **kwargs)
    gca().add_patch(c)
    axis([self.cut['x'][0], self.cut['x'][1], self.cut['y'][0], self.cut['y'][1]])

  def starshow(self, color='magenta', alpha=0.1):
    mask = self.cut.select(self.star['locations'])
    X = self.star['locations'][mask,0]
    Y = self.star['locations'][mask,1]
    plot(X, Y, ', ', color=color, alpha =alpha)
    axis([self.cut['x'][0], self.cut['x'][1], self.cut['y'][0], self.cut['y'][1]])

  def gasshow(self, component='mass', mode='mean|intensity', vmin=None, vmax=None, logscale=True, contour_levels=None, cmap=pygasmap, gamma=1.0, over=0.0, under=0.0):
    area = (self.cut.size[0] * (self.cut.size[1] * 1.0)/(self.shape[0] * self.shape[1]))
    print 'area of a pixel', area
    if component=='mass':
      cachename = component
      if not cachename in self.cache:
        self.cache[cachename] = zeros(dtype='f4', shape = (self.shape[0], self.shape[1]))
        self.raster(self.cache[cachename], component, quick=False)
      todraw = self.cache[cachename] / area
      todraw = todraw.T
    else:
      cachename = component + '_mean'
      if not cachename in self.cache:
        self.cache[cachename] = zeros(dtype='f4', shape = (self.shape[0], self.shape[1]))
        self.cache['mass'] = zeros(dtype='f4', shape = (self.shape[0], self.shape[1]))
        self.mraster(self.cache[cachename], self.cache['mass'], component, quick=False)
      mass = self.cache['mass']
      todraw = self.cache[cachename] / mass
      todraw = todraw.T
      mass = mass.T

    if logscale:
      todraw = log10(todraw)
    if vmin == None: vmin = fmin.reduce(todraw.flat)
    if vmax == None: vmax = fmax.reduce(todraw.flat)
    N = Normalize(vmin=vmin, vmax=vmax)
    image = cmap(N(todraw))
     
    if mode == 'intensity':
      weight = log10(mass)
      sort = argsort(weight.ravel())
      mmax = weight.ravel()[sort[int((len(sort) - 1 )* (1.0 - over))]]
      mmin = weight.ravel()[sort[int((len(sort) - 1 )* under)]]
      print mmax, mmin
      Nm = Normalize(vmax=mmax, vmin=mmin, clip=True)
      weight = Nm(weight) ** gamma
      image[:,:,0] = image[:,:,0] * weight
      image[:,:,1] = image[:,:,1] * weight
      image[:,:,2] = image[:,:,2] * weight

    print 'max, min =' , vmax, vmin
    if contour_levels != None:
      contour(todraw, extent=(self.cut['x'][0], self.cut['x'][1], self.cut['y'][0], self.cut['y'][1]), colors='k', linewidth=2, levels=contour_levels)
    imshow(image, origin='lower',
         extent=(self.cut['x'][0], self.cut['x'][1], self.cut['y'][0], self.cut['y'][1]), vmin=vmin, vmax=vmax, cmap=cmap)

  def velshow(self, relative=False, color='cyan', alpha=0.8):
    X,Y,vel = self.vector('vel', grids=(20,20), quick=False)
    mean = None
    if not type(relative) == bool or relative == True:
      if type(relative) == bool:
        mean = average(self.gas['vel'], weights=self.gas['mass'], axis=0)[0:2]
      else:
        mean = asarray(relative)
      vel[:,:,0]-=mean[0]
      vel[:,:,1]-=mean[1]
      print 'mean = ', mean
    print 'max component of velocity', abs(vel).max()
    sorted = absolute(vel.ravel())
    sorted.sort()
    scale = sorted[sorted.size * 9 / 10] * 20
    print X.shape, vel[:,:,0].shape
    quiver(X,Y, vel[:,:,0], vel[:,:,1], width=0.003, scale=scale, scale_units='width', angles='xy', color=color,alpha=alpha)
#  streamplot(X[0,:],Y[:,0], vel[:,:,0], vel[:,:,1], color=color)
    if mean != None: 
      quiver(self.cut.center[0], self.cut.center[1], mean[0], mean[1], scale=scale, scale_units='width', width=0.01, angles='xy', color=(0,0,0,0.5))
    axis([self.cut['x'][0], self.cut['x'][1], self.cut['y'][0], self.cut['y'][1]])

  def decorate(self, frameon=True):
    from mpl_toolkits.axes_grid.anchored_artists import AnchoredSizeBar
    ax = gca()
    if frameon :
      ticklabel_format(axis='x', useOffset=self.cut.center[0])
      ticklabel_format(axis='y', useOffset=self.cut.center[1])
      xticks(linspace(self.cut['x'][0], self.cut['x'][1], 5))
      yticks(linspace(self.cut['y'][0], self.cut['y'][1], 5))
    else :
      ax.axison = False

    l = (self.cut['x'][1] - self.cut['x'][0]) * 0.2
    l = l // 10 ** int(log10(l)) * 10 ** int(log10(l))
    if l > 500 :
      l/=1000.0
      l = int(l+0.5)
      text = r"%g Mpc/$h$" % l
      l *= 1000.0
    else:
      text = r"%g Kpc/$h$" %l
   
    b = AnchoredSizeBar(ax.transData, l, text, loc = 8, 
        pad=0.1, borderpad=0.5, sep=5, frameon=False)
    for r in b.size_bar.findobj(Rectangle):
      r.set_edgecolor('w')
    for t in b.txt_label.findobj(Text):
      t.set_color('w')
    ax.add_artist(b)

  def vector(self,component, grids=(20,20), quick=True):
    xs,xstep = linspace(self.cut['x'][0], self.cut['x'][1], grids[0], endpoint=False,retstep=True)
    ys,ystep = linspace(self.cut['y'][0], self.cut['y'][1], grids[1], endpoint=False,retstep=True)
    X,Y=meshgrid(xs+ xstep/2.0,ys+ystep/2.0)
    q = zeros(shape=(grids[0],grids[1],3), dtype='f4')
    mass = zeros(shape = (r.shape[0], r.shape[1]), dtype='f4')
    print 'num particles rastered', mraster(q, mass, component, quick)
    q[:,:,0]/=mass[:,:]
    q[:,:,1]/=mass[:,:]
    q[:,:,2]/=mass[:,:]
    return X,Y,q.transpose((1,0,2))

  def raster(self,r, component, quick=True):
    return rasterize(self.gas, r, component, xrange=self.cut[0], yrange=self.cut[1], zrange=self.cut[2], quick=quick)


  def mraster(self, r, mass, component, normed=True, quick=True):
    old = self.gas[component].copy()
    try:
      self.gas[component][:] *= self.gas['mass'][:]
    except:
      self.gas[component][:,0] *= self.gas['mass'][:]
      self.gas[component][:,1] *= self.gas['mass'][:]
      self.gas[component][:,2] *= self.gas['mass'][:]

    r.flat[:] = 0
    rt = rasterize(self.gas, [mass, r], ['mass', component], xrange=self.cut[0], yrange=self.cut[1], zrange=self.cut[2], quick=quick)
    self.gas[component] = old
    print old.max(), old.min()
    print self.gas.describe(component)

__default = Context()
gactx = __default
def redraw(func):
  def wrapped(*args, **kwargs):
    rt = func(*args, **kwargs)
    if isinteractive(): draw()
    return rt
  return wrapped

def use(*args, **kwargs): return __default.use(*args, **kwargs)
def read(*args, **kwargs): return __default.read(*args, **kwargs)

@redraw
def gasshow(*args, **kwargs): return __default.gasshow(*args, **kwargs)
@redraw
def velshow(*args, **kwargs): return __default.velshow(*args, **kwargs)
@redraw
def bhshow(*args, **kwargs): return __default.bhshow(*args, **kwargs)
@redraw
def decorate(*args, **kwargs): return __default.decorate(*args, **kwargs)
@redraw
def starshow(*args, **kwargs): return __default.starshow(*args, **kwargs)
@redraw
def circle(*args, **kwargs): return __default.circle(*args, **kwargs)

def rotate(*args, **kwargs): return __default.rotate(*args, **kwargs)
def get_redshift(): return __default.redshift
def get_gas(): return __default.gas
def get_bh(): return __default.bh
def set_resolution(res): __default.shape = (res,res)
def get_cut(): return __default.cut
def zoom(*args, **kwargs): return __default.zoom(*args, **kwargs)
def select(*args, **kwargs): return __default.select(*args, **kwargs)
def makeT():
  from gadget.constant.GADGET import TEMPERATURE_K
  from gadget.cosmology import default as _DC
  gas = get_gas()
  gas['T'] = zeros(dtype='f4', shape=gas.numpoints)
  _DC.ie2T(ie = gas['ie'], reh = gas['reh'], Xh = 0.76, out = gas['T'])
  gas['T'] *= TEMPERATURE_K

def Rvir(Mhalo):
  from gadget.cosmology import default as DC
  return DC.Rvir(Mhalo, z=get_redshift())

def Tvir(Mhalo):
  from gadget.constant.GADGET import TEMPERATURE_K
  from gadget.cosmology import default as DC
  return DC.Tvir(Mhalo, z=get_redshift()) * TEMPERATURE_K


def mergeBHs(threshold=1.0):
  bh = get_bh()
  if bh.numpoints == 0: return
  pos = bh['locations']
  posI = tile(pos, pos.shape[0]).reshape(pos.shape[0], pos.shape[0], 3)
  posJ = posI.transpose((1,0,2))
  d = posI - posJ
  d = sqrt((d ** 2).sum(axis=2))
  mask = d < threshold

  mergeinto = zeros(bh.numpoints, dtype='i4')
  for i in range(bh.numpoints): mergeinto[i] = nonzero(mask[i])[0][0]

  g, ind = unique(mergeinto, return_index=True)

  comp='bhmass'
  bak = bh[comp].copy()
  for i in range(bh.numpoints): bh[comp][i] = bak[mask[i]].sum()

  try :
    comp='bhmdot'
    bak = bh[comp].copy()
    for i in range(bh.numpoints): bh[comp][i] = bak[mask[i]].sum()
  except KeyError: pass


  comp='locations'
  #def sel(comp):
  bak = bh[comp].copy()
  for i in range(bh.numpoints): bh[comp][i] = bak[mask[i]][0]

  bh.numpoints = len(ind)
  for comp in bh: bh[comp] = bh[comp][ind]

