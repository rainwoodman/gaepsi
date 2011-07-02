#! python
from numpy import asarray, newaxis
from numpy import multiply
from matplotlib.pyplot import *
from gaepsi.constant import GADGET
from gaepsi.snapshot import Snapshot
from gaepsi.field import Field, Cut

from gaepsi.plot.image import rasterize
from gaepsi.plot.render import Colormap, color as gacolor
from gaepsi.tools.streamplot import streamplot
from gaepsi import ccode
from numpy import zeros, linspace, meshgrid, log10, average, vstack,absolute,fmin,fmax
from numpy import isinf, nan, argsort, isnan
from numpy import tile, unique, sqrt, nonzero
from numpy import ceil
from numpy import random
import matplotlib.pyplot as pyplot
gasmap = Colormap(levels =[0, 0.05, 0.2, 0.5, 0.6, 0.9, 1.0],
                      r = [0, 0.1 ,0.5, 1.0, 0.2, 0.0, 0.0],
                      g = [0, 0  , 0.2, 1.0, 0.2, 0.0, 0.0],
                      b = [0, 0  , 0  , 0.0, 0.4, 0.8, 1.0],
                      a = [1, 1,     1,   1,   1,   1,   1])
starmap = Colormap(levels =[0, 1.0],
                      r = [1, 1.0],
                      g = [1, 1.0],
                      b = [1, 1.0],
                      a = [1, 1.0])


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

pygascmap = pycmap(gasmap)
pystarcmap = pycmap(starmap)

def gacmap(pycmap):
  values = linspace(0, 1, 1024)
  colors = pycmap(values)
  return Colormap(levels = values, r = colors[:,0], g = colors[:, 1], b = colors[:,2], a = colors[:,3])

class GaplotContext:
  def __init__(self, shape = (600,600)):
    self.format = None
    self.shape = shape
    self.cache = {}
    self.masslimits = (None, None)
    self.F = {}

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
    self.F['gas'] = Field(components=self.components, cut=cut)
    self.F['bh'] = Field(components={'bhmass':'f4', 'bhmdot':'f4', 'id':'u8'}, cut=cut)
    self.F['star'] = Field(components={'sft':'f4', 'mass':'f4'}, cut=cut)
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
    return self.F['gas']

  @property
  def bh(self):
    return self.F['bh']

  @property
  def star(self):
    return self.F['star']

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
        self.bh.add_snapshot(snap, ptype = 5, components=['bhmass', 'bhmdot', 'id'])
      if use_star:
        self.star.add_snapshot(snap, ptype = 4, components=['mass', 'sft'])
    self.cache.clear()

  def radial_mean(self, component, bins=100, min=None, max=None):
    from numpy import histogram
    d = self.gas.dist(center=self.cut.center)
    if min is not None and max is not None: range=(min, max)
    else: range= None 
    mass, bins = histogram(d, range=range, bins=bins, weights=self.gas['mass'])
    value, bins = histogram(d, range=range, bins=bins, weights=self.gas['mass'] * self.gas[component])
    return bins[:-1], value/mass

  def rotate(self, *args, **kwargs):
    kwargs['center'] = self.cut.center
    self.gas.rotate(*args, **kwargs)
    self.bh.rotate(*args, **kwargs)
    self.star.rotate(*args, **kwargs)

    self.cache.clear()
 
  def vector(self, ftype, component, grids=(20,20), quick=True):
    xs,xstep = linspace(self.cut['x'][0], self.cut['x'][1], grids[0], endpoint=False,retstep=True)
    ys,ystep = linspace(self.cut['y'][0], self.cut['y'][1], grids[1], endpoint=False,retstep=True)
    X,Y=meshgrid(xs+ xstep/2.0,ys+ystep/2.0)
    q = zeros(shape=(grids[0],grids[1],3), dtype='f4')
    mass = zeros(shape = (q.shape[0], q.shape[1]), dtype='f4')
    print 'num particles rastered', self.mraster(q, ftype, mass, component, quick)
    q[:,:,0]/=mass[:,:]
    q[:,:,1]/=mass[:,:]
    q[:,:,2]/=mass[:,:]
    return X,Y,q.transpose((1,0,2))

  def raster(self, ftype, component, quick=True, cache=True):
    """ ftype is amongst 'gas', 'star', those with has an sml  """
    cachename = ftype + '_' + component
    field = self.F[ftype]
    if cache and cachename in self.cache:
      return self.cache[cachename].copy()
    result = zeros(dtype='f4', shape = (self.shape[0], self.shape[1]))
    rasterize(field, result, component, xrange=self.cut[0], yrange=self.cut[1], zrange=self.cut[2], quick=quick)
    if cache : 
      self.cache[cachename] = result
      return result.copy()
    else:
      return result

  def mraster(self, ftype, component, normed=True, quick=True, cache=True):
    cachename = ftype + '_' + component + '_mass'
    if cache and cachename in self.cache:
      return self.cache[ftype + '_' + 'mass'].copy(), self.cache[cachename].copy()

    field = self.F[ftype]
    old = field[component].copy()
    try:
      if len(field[component].shape) == 1:
        field[component][:] *= field['mass'][:]
      else:
        field[component][:,0] *= field['mass'][:]
        field[component][:,1] *= field['mass'][:]
        field[component][:,2] *= field['mass'][:]
    
      result = zeros(dtype='f4', shape = (self.shape[0], self.shape[1]))
      mass = zeros(dtype='f4', shape = (self.shape[0], self.shape[1]))
      rasterize(field, [mass, result], ['mass', component], xrange=self.cut[0], yrange=self.cut[1], zrange=self.cut[2], quick=quick)
      if cache:
        self.cache[ftype + '_' + 'mass'] = mass
        self.cache[cachename] = result
        return mass.copy(), result.copy()
      else:
        return mass, result
    except Exception as e:
      raise e
    finally:
      field[component] = old

  def imfield(self, ftype, component='mass', mode='mean|weight|intensity'):
    """raster a field. ftype can be gas or star. for mass component, the mean density per area is plotted. for other components, mode determines
       when mode==mean, the mass weighted mean is plotted.
       when mode==weight, the mass weighted sum is plotted.
       when mode==intensity, the mass weighted mean is plotted, but the luminosity is reduced according to the log of the mass."""
    if component=='mass':
      todraw = self.raster(ftype, component, quick=False)
      todraw /= self.pixel_area
      print 'area of a pixel', self.pixel_area
      return todraw, None
    else:
      mass,todraw = self.mraster(ftype, component, quick=False)
      if mode != 'weight':
        todraw /= mass
      return todraw, mass

  def bhshow(self, ax, radius=4, labelfmt=None, vmin=None, vmax=None, count=-1, *args, **kwargs):
    from matplotlib.collections import CircleCollection
    mask = self.cut.select(self.bh['locations'])
    X = self.bh['locations'][mask,0]
    Y = self.bh['locations'][mask,1]
    ID = self.bh['id'][mask]
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
    if count > 0: 
      ind = (-R).argsort()
      X = X[ind[0:count]]
      Y = Y[ind[0:count]]
      R = R[ind[0:count]]
      ID = ID[ind[0:count]]

    R*=radius**2
    print R.min(), R.max()
#    R.clip(min=4, max=radius**2)
    if not 'edgecolor' in kwargs:
      kwargs['edgecolor'] = 'green'
    if not 'facecolor' in kwargs:
      kwargs['facecolor'] = (0, 1, 0, 0.0)
    ax.scatter(X,Y, s=R*radius, marker='o', **kwargs)
    if labelfmt: 
      for x,y,id in zip(X,Y,ID):
        rat = random.random() * 360
        if rat > 90 and rat <= 180:
          trat = rat + 180
        elif rat > 180 and rat <=270:
          trat = rat - 180
        else:
          trat = rat
        if rat < 315 and rat > 135: 
          dir = 1
          rat -= 180
        else: 
          dir = 0
        ax.text(x,y, labelfmt % id, withdash=True, 
          dashlength=radius * 10,
          dashpush=radius,
          dashdirection=dir,
          rotation=trat,
          dashrotation=rat,
          color='white'
          )

  #  col = CircleCollection(offsets=zip(X.flat,Y.flat), sizes=(R * radius)**2, edgecolor='green', facecolor='none', transOffset=gca().transData)
  #  ax.add_collection(col)

  def starshow_poor(self, ax, *args, **kwargs):
    star = self.star
    mask = self.cut.select(star['locations'])
    X = star['locations'][mask,0]
    Y = star['locations'][mask,1]
    if not 'color' in kwargs:
      kwargs['color'] = 'white'
    if not 'alpha' in kwargs:
      kwargs['alpha'] = 0.5
    ax.plot(X, Y, ', ', *args, **kwargs)

  def reset_view(self, ax):
    left,right =self.cut['x']
    bottom, top = self.cut['y']
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)

  def makeT(self):
    """T will be in Kelvin"""
    gas =self.gas
    C = gas.cosmology
    gas['T'] = zeros(dtype='f4', shape=gas.numpoints)
    C.ie2T(ie = gas['ie'], ye = gas['ye'], Xh = 0.76, out = gas['T'])
    gas['T'] *= C.units.TEMPERATURE

  def mergeBHs(self, threshold=1.0):
    bh = self.bh
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

  def mlim(self, vmin=None, vmax=None):
    """ if there isn't a mass limit, calculate one from gas
        if there is one specified in parameters, override the
          current limit
        use 'auto' 'auto' to reset to the auto calculated limits
        returns the new limits.
    """
    if vmin is None:
      vmin = self.masslimits[0]
    if vmin is None:
      vmin = 'auto'
    if vmax is None:
      vmax = self.masslimits[1]
    if vmax is None:
      vmax = 'auto'
    if vmin == 'auto' or vmax == 'auto':
      if self.gas.numpoints == 0:
        raise Exception("Cannot calculate mlim without a gas field")
      mass = self.raster('gas', 'mass', quick=False)
      mass /= self.pixel_area
      if vmin == 'auto':
        vmin = ccode.pmin.reduce(mass.flat)
      if vmax == 'auto':
        vmax = fmax.reduce(mass.flat)

    self.masslimits = (vmin, vmax)
    return self.masslimits

  def fieldshow(self, ax_or_fig, ftype, component='mass', mode='mean|weight|intensity', vmin=None, vmax=None, logscale=True, cmap=pygascmap, gamma=1.0, over=None, under=None, return_raster=False, levels=None):
    todraw, mass = self.imfield(ftype=ftype, component=component, mode=mode)

    if logscale:
     if vmin is not None:
       vmin = 10 ** vmin
     if vmax is not None:
       vmax = 10 ** vmax

    if component == 'mass':
      if vmin is None: vmin = 'auto'
      if vmax is None: vmax = 'auto'
      vmin, vmax = self.mlim(vmin, vmax)
    else:
      if vmin is None: 
        if logscale:
          vmin = ccode.pmin.reduce(todraw.flat)
        else:
          vmin = fmin.reduce(todraw.flat)
      if vmax is None: vmax = fmax.reduce(todraw.flat)

    if logscale:
      log10(todraw, todraw)
      vmin = log10(vmin)
      vmax = log10(vmax)

    image = zeros(dtype = ('u1', 4), shape = todraw.shape)
    # gacolor is much faster then matplotlib's normalize and uses less memory(4times fewer).
    gacolor(image, todraw, max = vmax, min = vmin, logscale=False, colormap = gacmap(cmap))
    if mode == 'intensity':
      weight = mass
      log10(weight, weight)
      weight[isinf(weight)] = nan
      weight[isnan(weight)] = fmin.reduce(weight.ravel())
      sort = argsort(weight.ravel())
      mmin, mmax = log10(self.mlim())
      if over is not None:
        mmax = weight.ravel()[sort[int((len(sort) - 1 )* (1.0 - over))]]
      if under is not None:
        mmin = weight.ravel()[sort[int((len(sort) - 1 )* under)]]
      print mmax, mmin
      Nm = Normalize(vmax=mmax, vmin=mmin, clip=True)
      weight = Nm(weight) ** gamma
      multiply(image[:,:,3], weight[:, :, ], image[:,:,3])
    print 'max, min =' , vmax, vmin
    if return_raster: return (image, vmin, vmax, todraw)

    if hasattr(ax_or_fig, 'figimage'):
      ret = ax_or_fig.figimage(image.transpose((1,0,2)), 0, 0, origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
    else:
      ret = ax_or_fig.imshow(image.transpose((1,0,2)), origin='lower',
         extent=self.extent, vmin=vmin, vmax=vmax, cmap=cmap)
    if levels is not None:
      if hasattr(ax_or_fig, 'contour'):
        ax_or_fig.contour(todraw.T, extent=self.extent, colors='k', linewidth=2, levels=levels)
      else:
        ax_or_fig.gca().contour(todraw.T, extent=self.extent, colors='k', linewidth=2, levels=levels)
    return ret

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

  def bhshow(self, *args, **kwargs):
    self.gaplot.bhshow(self.gca(), *args, **kwargs)
  bhshow.__doc__ = GaplotContext.bhshow.__doc__

  def circle(self, *args, **kwargs):
    from matplotlib.patches import Circle
    c = Circle(*args, **kwargs)
    self.gca().add_patch(c)

  def reset_view(self):
    self.gaplot.reset_view(self.gca())

  def mlim(self, *args, **kwargs):
    return self.gaplot.mlim(*args, **kwargs)

  def gasshow(self, component='mass', use_figimage=False, *args, **kwargs):
    "see fieldshow for docs"
    kwargs['component'] = component
    if use_figimage:
      ret = self.gaplot.fieldshow(self, 'gas', *args, **kwargs)
    else:
      ret = self.gaplot.fieldshow(self.gca(), 'gas', *args, **kwargs)
    self.gca()._sci(ret)

  def starshow(self, component='mass', use_figimage=False, *args, **kwargs):
    "see fieldshow for docs"
    kwargs['component'] = component
    if use_figimage:
      ret = self.gaplot.fieldshow(self, 'star', *args, **kwargs)
    else:
      ret = self.gaplot.fieldshow(self.gca(), 'star', *args, **kwargs)
    self.gca()._sci(ret)

  def starshow_poor(self, *args, **kwargs):
     self.gaplot.starshow_poor(self.gca(), *args, **kwargs)

  def velshow(self, ftype, relative=False, color='cyan', alpha=0.8):
    X,Y,vel = self.gaplot.vector(ftype, 'vel', grids=(20,20), quick=False)
    field = self.F[ftype]
    mean = None
    if not type(relative) == bool or relative == True:
      if type(relative) == bool:
        mean = average(field['vel'], weights=field['mass'], axis=0)[0:2]
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
    ax.set_axis_bgcolor('k')
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

methods = ['use', 'gasshow', 'bhshow', 'read', 'starshow', 'starshow_poor', 'decorate', 'unfold', 'zoom', 'drawscale', 'circle', 'reset_view', 'rotate', 'mlim']
import sys
__module__ = sys.modules[__name__]
def __mkfunc(method):
  mbfunc = GaplotFigure.__dict__[method]
  def func(*args, **kwargs):
    f = gcf()
    if not hasattr(f, 'gaplot'):
      setattr(f, 'gaplot', GaplotContext())
    rt = mbfunc(gcf() , *args, **kwargs)
    if isinteractive(): draw()
    return rt
  func.__doc__ = mbfunc.__doc__
  return func
for method in methods:
  __module__.__dict__[method] = __mkfunc(method)

#def figure(*args, **kwargs):
#  kwargs['FigureClass'] = GaplotFigure
#  return pyplot.figure(*args, **kwargs)


