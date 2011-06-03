#! python
from numpy import asarray, newaxis
from numpy import multiply
from matplotlib.pyplot import *
from gadget.snapshot import Snapshot
from gadget.field import Field, Cut

from gadget.plot.image import rasterize
from gadget.plot.render import Colormap, color as gacolor
from gadget.tools.streamplot import streamplot
from gadget import ccode
from numpy import zeros, linspace, meshgrid, log10, average, vstack,absolute,fmin,fmax
from numpy import isinf, nan, argsort
from numpy import tile, unique, sqrt, nonzero
from numpy import ceil
import matplotlib.pyplot as pyplot
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


def pycmap(gacmap):
  from matplotlib.colors import ListedColormap
  r = gacmap.table['r']
  g = gacmap.table['g']
  b = gacmap.table['b']
  rt = ListedColormap(vstack((r, g, b)).T)
  rt.set_over((r[-1], g[-1], b[-1]))
  rt.set_under((r[0], g[0], b[0]))
  rt.set_bad(color='k', alpha=0)
  return rt

pygasmap = pycmap(gasmap)

def gacmap(pycmap):
  values = linspace(0, 1, 1024)
  colors = pycmap(values)
  return Colormap(levels = values, r = colors[:,0], g = colors[:, 1], b = colors[:,2], a = colors[:,3])

class GaplotContext:
  def __init__(self, shape = (1000,1000)):
    self.format = None
    self.shape = shape
    self.cache = {}

  def zoom(self, center=None, size=None):
    if center == None:
      center = self.cut.center
    if size == None:
      size = self.cut.size
    self.cut=Cut(center=center, size=size)
    self.cache.clear()

  def unfold(self, M):
    self.gas.unfold(M)
    self.star.unfold(M)
    self.bh.unfold(M)
    self.cut = Cut(xcut=[0, self.gas.boxsize[0]], 
                   ycut=[0, self.gas.boxsize[1]], 
                   zcut=[0, self.gas.boxsize[2]])
  @property
  def pixel_area(self):
    return (self.cut.size[0] * (self.cut.size[1] * 1.0)/(self.shape[0] * self.shape[1]))

  def select(self, mask):
    self.gas.set_mask(mask)
    self.cache.clear()

  def use(self, snapname, format, components={}, cut=None):
    self.components = components
    self.components['mass'] = 'f4'
    self.components['sml'] = 'f4'
    self.components['vel'] = ('f4', 3)
    self.snapname = snapname
    self.format = format
    self.__gas = Field(components=self.components, cut=cut)
    self.__bh = Field(components={'bhmass':'f4'}, cut=cut)
    self.__star = Field(components={'sft':'f4'}, cut=cut)
    try:
      snapname = self.snapname % 0
    except TypeError:
      snapname = self.snapname
    snap = Snapshot(snapname, self.format)
    self.gas.init_from_snapshot(snap)
    self.bh.init_from_snapshot(snap)
    self.star.init_from_snapshot(snap)
    self.C = snap.C.copy()
    if cut == None:
      self.cut = Cut(xcut=[0, snap.C['L']], ycut=[0, snap.C['L']], zcut=[0, snap.C['L']])
    else:
      self.cut = cut
    self.cache.clear()
    

  @property
  def extent(self):
    return (self.cut['x'][0], self.cut['x'][1], self.cut['y'][0], self.cut['y'][1])
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

  def read(self, fids=None, use_gas=True, use_bh=True, use_star=True):
    if fids != None:
      snapnames = [self.snapname % i for i in fids]
    else:
      snapnames = [self.snapname]

    for snapname in snapnames:
      snap = Snapshot(snapname, self.format)
      if use_gas:
        self.gas.add_snapshot(snap, ptype = 0, components=self.components.keys())
        print snapname , 'loaded', 'gas particles', self.gas.numpoints
      if use_bh:
        self.bh.add_snapshot(snap, ptype = 5, components=['bhmass'])
      if use_star:
        self.star.add_snapshot(snap, ptype = 4, components=['sft'])
    self.cache.clear()

  def rotate(self, *args, **kwargs):
    self.gas.rotate(*args, **kwargs)
    self.bh.rotate(*args, **kwargs)
    self.star.rotate(*args, **kwargs)

    self.cache.clear()
 
  def vector(self,component, grids=(20,20), quick=True):
    xs,xstep = linspace(self.cut['x'][0], self.cut['x'][1], grids[0], endpoint=False,retstep=True)
    ys,ystep = linspace(self.cut['y'][0], self.cut['y'][1], grids[1], endpoint=False,retstep=True)
    X,Y=meshgrid(xs+ xstep/2.0,ys+ystep/2.0)
    q = zeros(shape=(grids[0],grids[1],3), dtype='f4')
    mass = zeros(shape = (q.shape[0], q.shape[1]), dtype='f4')
    print 'num particles rastered', self.mraster(q, mass, component, quick)
    q[:,:,0]/=mass[:,:]
    q[:,:,1]/=mass[:,:]
    q[:,:,2]/=mass[:,:]
    return X,Y,q.transpose((1,0,2))

  def raster(self, component, quick=True, cache=True):
    cachename = component
    if cache and cachename in self.cache:
      return self.cache[cachename].copy()
    result = zeros(dtype='f4', shape = (self.shape[0], self.shape[1]))
    rasterize(self.gas, result, component, xrange=self.cut[0], yrange=self.cut[1], zrange=self.cut[2], quick=quick)
    if cache : 
      self.cache[cachename] = result
      return result.copy()
    else:
      return result

  def mraster(self, component, normed=True, quick=True, cache=True):
    cachename = component + '_mass'
    if cache and cachename in self.cache:
      return self.cache['mass'].copy(), self.cache[cachename].copy()

    old = self.gas[component].copy()
    try:
      self.gas[component][:] *= self.gas['mass'][:]
    except:
      self.gas[component][:,0] *= self.gas['mass'][:]
      self.gas[component][:,1] *= self.gas['mass'][:]
      self.gas[component][:,2] *= self.gas['mass'][:]

    
    result = zeros(dtype='f4', shape = (self.shape[0], self.shape[1]))
    mass = zeros(dtype='f4', shape = (self.shape[0], self.shape[1]))
    rasterize(self.gas, [mass, result], ['mass', component], xrange=self.cut[0], yrange=self.cut[1], zrange=self.cut[2], quick=quick)
    self.gas[component] = old
    if cache:
      self.cache['mass'] = mass
      self.cache[cachename] = result
      return mass.copy(), result.copy()
    else:
      return mass, result

  def imgas(self, component='mass', mode='mean|intensity', vmin=None, vmax=None, logscale=True, cmap=pygasmap, gamma=1.0, over=0.0, under=0.0):
    if component=='mass':
      todraw = self.raster(component, quick=False)
      todraw /= self.pixel_area
      print 'area of a pixel', self.pixel_area
    else:
      mass,todraw = self.mraster(component, quick=False)
      todraw /= mass

    if logscale:
      log10(todraw, todraw)
    if vmin == None: vmin = fmin.reduce(todraw.flat)
    if vmax == None: vmax = fmax.reduce(todraw.flat)
#    N = Normalize(vmin=vmin, vmax=vmax)
    image = zeros(dtype = ('u1', 3), shape = todraw.shape)
    gacolor(image, todraw, max = vmax, min = vmin, logscale=False, colormap = gacmap(cmap))
#    image = cmap(N(todraw))
     
    if mode == 'intensity':
      weight = mass
      log10(weight, weight)
      sort = argsort(weight.ravel())
      mmax = weight.ravel()[sort[int((len(sort) - 1 )* (1.0 - over))]]
      mmin = weight.ravel()[sort[int((len(sort) - 1 )* under)]]
      print mmax, mmin
      Nm = Normalize(vmax=mmax, vmin=mmin, clip=True)
      weight = Nm(weight) ** gamma
      multiply(image[:,:,0:3], weight[:, :, newaxis], image[:,:,0:3])
    print 'max, min =' , vmax, vmin
    return image

  def bhshow(self, ax, radius, vmin=None, vmax=None, count=-1, *args, **kwargs):
    from matplotlib.collections import CircleCollection
    mask = self.cut.select(self.bh['locations'])
    X = self.bh['locations'][mask,0]
    Y = self.bh['locations'][mask,1]
    bhmass = self.bh['bhmass'][mask]
    if bhmass.size == 0: return
    if vmax is None:
      vmax = bhmass.max()
    if vmin is None:
      vmin = bhmass.min()
    print 'bhshow, vmax, vmin =', vmax, vmin
    R = log10(bhmass)
    Nm = Normalize(vmax=log10(vmax), vmin=log10(vmin), clip=True)
    R = Nm(R)
    ind = (-R).argsort()
    X = X[ind[0:count]]
    Y = Y[ind[0:count]]
    R = R[ind[0:count]]
    R*=radius**2
    print R.min(), R.max()
#    R.clip(min=4, max=radius**2)
    if not 'edgecolor' in kwargs:
      kwargs['edgecolor'] = 'green'
    if not 'facecolor' in kwargs:
      kwargs['facecolor'] = (0, 1, 0, 0.0)
    ax.scatter(X,Y, s=R*radius, marker='o', **kwargs)
  #  col = CircleCollection(offsets=zip(X.flat,Y.flat), sizes=(R * radius)**2, edgecolor='green', facecolor='none', transOffset=gca().transData)
  #  ax.add_collection(col)
  def starshow(self, ax, *args, **kwargs):
    star = self.star
    mask = self.cut.select(star['locations'])
    X = star['locations'][mask,0]
    Y = star['locations'][mask,1]
    if not 'color' in kwargs:
      kwargs['color'] = 'white'
    if not 'alpha' in kwargs:
      kwargs['alpha'] = 0.5
    ax.plot(X, Y, ', ', *args, **kwargs)

class GaplotFigure(Figure):
  def __init__(self, gaplot_context=None, *args, **kwargs):
    if gaplot_context != None:
      self.gaplot = gaplot_context
    else:
      self.gaplot = GaplotContext()
    Figure.__init__(self, *args, **kwargs)

  def read(self, *args, **kwargs):
    return self.gaplot.read(*args, **kwargs)
  def use(self, *args, **kwargs):
    return self.gaplot.use(*args, **kwargs)
  def zoom(self, *args, **kwargs):
    return self.gaplot.zoom(*args, **kwargs)
  def unfold(self, *args, **kwargs):
    return self.gaplot.unfold(*args, **kwargs)
  def rotate(self, *args, **kwargs):
    return self.gaplot.rotate(*args, **kwargs)

  def reset_view(self):
    left,right =self.gaplot.cut['x']
    bottom, top = self.gaplot.cut['y']
    self.gca().set_xlim(left, right)
    self.gca().set_ylim(bottom, top)

  def bhshow(self, *args, **kwargs):
    self.gaplot.bhshow(self.gca(), *args, **kwargs)
    self.reset_view()
  bhshow.__doc__ = GaplotContext.bhshow.__doc__

  def circle(self, *args, **kwargs):
    from matplotlib.patches import Circle
    c = Circle(*args, **kwargs)
    self.gca().add_patch(c)
    self.reset_view()

  def starshow(self, *args, ** kwargs):
    self.gaplot.starshow(self.gca(), *args, **kwargs)
    self.reset_view()

  def gasshow(self, use_figimage=False, *args, **kwargs):
    image = self.gaplot.imgas(*args, **kwargs)
#    if contour_levels != None:
#      self.gca().contour(todraw.T, extent=self.gaplot.extent, colors='k', linewidth=2, levels=contour_levels)
    try: vmin = kwargs['vmin']
    except: vmin = None
    try: vmax = kwargs['vmax']
    except: vmax = None
    cmap = kwargs['cmap']
    if use_figimage: 
      self.figimage(image.transpose((1,0,2)), 0, 0, origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
    else:
      self.gca().imshow(image.transpose((1,0,2)), origin='lower',
         extent=self.gaplot.extent, vmin=vmin, vmax=vmax, cmap=cmap)

  def velshow(self, relative=False, color='cyan', alpha=0.8):
    X,Y,vel = self.gaplot.vector('vel', grids=(20,20), quick=False)
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
    self.gca().quiver(X,Y, vel[:,:,0], vel[:,:,1], width=0.003, scale=scale, scale_units='width', angles='xy', color=color,alpha=alpha)
#  streamplot(X[0,:],Y[:,0], vel[:,:,0], vel[:,:,1], color=color)
    if mean != None: 
      self.gca().quiver(self.gaplot.cut.center[0], self.gaplot.cut.center[1], mean[0], mean[1], scale=scale, scale_units='width', width=0.01, angles='xy', color=(0,0,0,0.5))
    self.reset_view()

  def drawscale(self):
    from mpl_toolkits.axes_grid.anchored_artists import AnchoredSizeBar
    ax = self.gca()
    l = (self.gaplot.cut.size[0]) * 0.2
    l = l // 10 ** int(log10(l)) * 10 ** int(log10(l))
    if l > 500 :
      l/=1000.0
      l = int(l+0.5)
      text = r"%g Mpc/h" % l
      l *= 1000.0
    else:
      text = r"%g Kpc/h" %l
   
    b = AnchoredSizeBar(ax.transData, l, text, loc = 8, 
        pad=0.1, borderpad=0.5, sep=5, frameon=False)
    for r in b.size_bar.findobj(Rectangle):
      r.set_edgecolor('w')
    for t in b.txt_label.findobj(Text):
      t.set_color('w')
    ax.add_artist(b)


  def decorate(self, frameon=True, titletext=None):
    ax = self.gca()
    cut = self.gaplot.cut
    if frameon :
      ax.ticklabel_format(axis='x', useOffset=cut.center[0])
      ax.ticklabel_format(axis='y', useOffset=cut.center[1])
      ax.set_xticks(linspace(cut['x'][0], cut['x'][1], 5))
      ax.set_yticks(linspace(cut['y'][0], cut['y'][1], 5))
      ax.set_title(titletext)
    else :
      ax.axison = False
      if(titletext !=None):
        ax.text(0.1, 0.9, titletext, fontsize='small', color='white', transform=ax.transAxes)

methods = ['use', 'gasshow', 'bhshow', 'read', 'starshow', 'decorate', 'unfold']
import sys
__module__ = sys.modules[__name__]
def __mkfunc(method):
  mbfunc = GaplotFigure.__dict__[method]
  def func(*args, **kwargs):
    rt = mbfunc(gcf() , *args, **kwargs)
    if isinteractive(): draw()
    return rt
  func.__doc__ = mbfunc.__doc__
  return func
for method in methods:
  __module__.__dict__[method] = __mkfunc(method)

def figure(*args, **kwargs):
  kwargs['FigureClass'] = GaplotFigure
  return pyplot.figure(*args, **kwargs)

def makeT(gas):
  from gadget.constant.GADGET import TEMPERATURE_K
  from gadget.cosmology import default as _DC
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


def mergeBHs(bh, threshold=1.0):
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

