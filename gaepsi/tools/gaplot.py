#! python
from warnings import warn
warn("module tools.gaplot is deprecated")
from numpy import asarray, newaxis
from numpy import multiply, divide, add, array
from numpy import max, min, isscalar
from matplotlib.pyplot import *
from gaepsi.snapshot import Snapshot
from gaepsi.field import Field, Cut
from gaepsi.tools.keyframes import KeyFrames

from meshmap import Meshmap

from gaepsi.tools.streamplot import streamplot
from gaepsi.tools.spikes import SpikeCollection

from gaepsi import ccode

from numpy import zeros, linspace, meshgrid, log10, average, vstack,absolute,fmin,fmax, ones
from numpy import isinf, nan, argsort, isnan, inf
from numpy import float32
from numpy import tile, unique, sqrt, nonzero
from numpy import ceil
from numpy import random
import matplotlib.pyplot as pyplot
from Queue import Queue
from gaepsi.tools.simplegl import Camera
from gaepsi.tools import sharedmem

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
gasmap = Colormap(levels =[0, 0.05, 0.2, 0.5, 0.6, 0.9, 1.0],
                      r = [0, 0.1 ,0.5, 1.0, 0.2, 0.0, 0.0],
                      g = [0, 0  , 0.2, 1.0, 0.2, 0.0, 0.0],
                      b = [0, 0  , 0  , 0.0, 0.4, 0.8, 1.0],
                      a = [1, 1,     1,   1,   1,   1,   1])
starmap = Colormap(levels =[0, 1.0],
                      r = [1, 1.0],
                      g = [1, 1.0],
                      b = [1, 1.0],
                      a = [0, 1.0])

def gacmap(pycmap):
  values = linspace(0, 1, 1024)
  colors = pycmap(values)
  return Colormap(levels = values, r = colors[:,0], g = colors[:, 1], b = colors[:,2], a = colors[:,3])

def pycmap(gacmap):
  from matplotlib.colors import ListedColormap
  from numpy import vstack
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

class GaplotContext(object):
  def __init__(self, shape = (600,600)):
    self.format = None
    self.shape = shape
    self.cache1 = {}
    self.cache2 = {}
    self.F = {}
    self.cut = Cut()
    self.periodic = False

  def invalidate(self):
    self.cache1.clear()
    self.cache2.clear()
   
  def zoom(self, center=None, size=None):
    if center is None:
      center = self.cut.center
    if size is None:
      size = self.cut.size
    self.cut.take(Cut(center=center, size=size))
    self.invalidate()

  def autozoom(self, ftype):
    if self.F[ftype].numpoints > 0:
      min=self.F[ftype]['locations'].min(axis=0)
      max=self.F[ftype]['locations'].max(axis=0)
      self.zoom(center=(min+max) * 0.5, size=(max - min))

  def slice(self, z, thickness):
    cut = Cut(center=self.cut.center, size=self.cut.size)
    cut['z'] = [z - thickness / 2.0, z + thickness / 2.0]
    self.cut.take(cut)
    self.invalidate()

  def unfold(self, M):
    self.gas.unfold(M, self.boxsize)
    self.star.unfold(M, self.boxsize)
    self.boxsize = self.bh.unfold(M, self.boxsize)
    self.cut.take(Cut(xcut=[0, self.boxsize[0]], 
                   ycut=[0, self.boxsize[1]], 
                   zcut=[0, self.boxsize[2]]))
  @property
  def pixel_area(self):
    return (self.cut.size[0] * (self.cut.size[1] * 1.0)/(self.shape[0] * self.shape[1]))

  def use(self, snapname, format, components={}, 
          bhcomponents={'bhmass':'f4', 'bhmdot':'f4', 'id':'u8'}, 
          starcomponents={'sft':'f4', 'mass':'f4'}, gas=0, halo=1, disk=2, bulge=3, star=4, bh=5, cut=None, gascomponents=None, periodic=True):
    if gascomponents is not None:
      self.components = gascomponents
    else:
      self.components = components
      self.components['mass'] = 'f4'
      self.components['sml'] = 'f4'

    self.snapname = snapname
    self.format = format
    self.F['gas'] = Field(components=self.components)
    self.F['bh'] = Field(components=bhcomponents)
    self.F['star'] = Field(components=starcomponents)
    self.F['halo'] = Field(components=self.components)
    self.F['disk'] = Field(components=self.components)
    self.F['bulge'] = Field(components=self.components)

    self.ptype = {
      "gas": gas,
      "halo": halo,
      "disk": disk,
      "bulge": bulge,
      "star": star,
      "bh": bh,
    }
    try:
      snapname = self.snapname % 0
    except TypeError:
      snapname = self.snapname
    snap = Snapshot(snapname, self.format)
    self.gas.init_from_snapshot(snap)
    self.bh.init_from_snapshot(snap)
    self.star.init_from_snapshot(snap)
    self.halo.init_from_snapshot(snap)
    self.disk.init_from_snapshot(snap)
    self.bulge.init_from_snapshot(snap)

    self.C = snap.C
    if cut is not None:
      self.cut.take(cut)
    else:
      try:
        boxsize = snap.C['boxsize']
        self.cut.take(Cut(xcut=[0, boxsize], ycut=[0, boxsize], zcut=[0, boxsize]))
      except:
        pass

    self.boxsize = ones(3) * snap.C['boxsize']
    self.redshift = snap.C['redshift']
    self.invalidate()
    self.periodic = periodic
 
  @property
  def extent(self):
    return (self.cut['x'][0], self.cut['x'][1], self.cut['y'][0], self.cut['y'][1])

  @property
  def halo(self):
    return self.F['halo']
  @halo.setter
  def halo(self, value):
    self.F['halo'] = value
    print 'halo set'
    self.invalidate()

  @property
  def bulge(self):
    return self.F['bulge']
  @bulge.setter
  def bulge(self, value):
    self.F['bulge'] = value
    print 'bulge set'
    self.invalidate()
  @property
  def disk(self):
    return self.F['disk']
  @disk.setter
  def disk(self, value):
    self.F['disk'] = value
    print 'disk set'
    self.invalidate()

  @property
  def gas(self):
    return self.F['gas']
  @gas.setter
  def gas(self, value):
    self.F['gas'] = value
    print 'gas set'
    self.invalidate()

  @property
  def bh(self):
    return self.F['bh']
  @bh.setter
  def bh(self, value):
    self.F['bh'] = value
    print 'bh set'
    self.invalidate()
  @property
  def star(self):
    return self.F['star']
  @star.setter
  def star(self, value):
    self.F['star'] = value
    print 'star set'
    self.invalidate()

  def read(self, fids=None, use_gas=True, use_bh=True, use_star=True, use_halo=False, use_disk=False, use_bulge=False, numthreads=None):
    if fids is not None:
      snapnames = [self.snapname % i for i in fids]
    elif '%d' in self.snapname:
      snapnames = [self.snapname % i for i in range(self.C['Nfiles'])]
    else:
      snapnames = [self.snapname]
    snapshots = [Snapshot(snapname, self.format) for snapname in snapnames]

    if use_gas:
      self.gas.take_snapshots(snapshots, ptype = self.ptype['gas'], nthreads=numthreads, cut=self.cut)
    if use_halo:
      self.halo.take_snapshots(snapshots, ptype = self.ptype['halo'], nthreads=numthreads, cut=self.cut)
    if use_disk:
      self.disk.take_snapshots(snapshots, ptype = self.ptype['disk'], nthreads=numthreads, cut=self.cut)
    if use_bulge:
      self.bulge.take_snapshots(snapshots, ptype = self.ptype['bulge'], nthreads=numthreads, cut=self.cut)
    if use_bh:
      self.bh.take_snapshots(snapshots, ptype = self.ptype['bh'], nthreads=numthreads, cut=self.cut)
    if use_star:
      self.star.take_snapshots(snapshots, ptype = self.ptype['star'], nthreads=numthreads, cut=self.cut)

    self.invalidate()


  def radial_mean(self, component, weightcomponent='mass', bins=100, min=None, max=None, std=False, origin=None):
    from numpy import histogram
    if origin is None: origin = self.cut.center
    d = self.gas.dist(origin=origin)
    print 'radial_mean', origin
    if min is not None and max is not None: range=(min, max)
    else: range= None 
    if weightcomponent is not None:
      m = self.gas[weightcomponent]
      mx = m * self.gas[component]
      mass, bins = histogram(d, range=range, bins=bins, weights=m)
      value, bins = histogram(d, range=range, bins=bins, weights=mx)
      value /= mass
      if std:
        mxx = mx * self.gas[component]
        std, bins = histogram(d, range=range, bins=bins, weights=mxx)
        std = sqrt(abs(std / mass - value ** 2))
        return bins[:-1], value, std
      return bins[:-1], value
    else:
      mx = self.gas[component]
      value, bins = histogram(d, range=range, bins=bins, weights=mx)
      value = value.cumsum()
      r2,r1 = bins[1:], bins[:-1]
      r = .5 * (r1 + r2)
      V = 4 / 3.0 * 3.14 * (r ** 3)
      value /= V
      if std:
# possion error
        std, bins = histogram(d, range=range, bins=bins)
        return bins[:-1], value, value * sqrt(std) / std
      return bins[:-1], value

  def rotate(self, *args, **kwargs):
    kwargs['origin'] = self.cut.center
    self.gas.rotate(*args, **kwargs)
    self.bh.rotate(*args, **kwargs)
    self.star.rotate(*args, **kwargs)
    self.invalidate()

 
  def vector(self, ftype, component, grids=(20,20), quick=True):
    xs,xstep = linspace(self.cut['x'][0], self.cut['x'][1], grids[0], endpoint=False,retstep=True)
    ys,ystep = linspace(self.cut['y'][0], self.cut['y'][1], grids[1], endpoint=False,retstep=True)
    X,Y=meshgrid(xs+ xstep/2.0,ys+ystep/2.0)
    q = zeros(shape=(grids[0],grids[1],3), dtype='f4')
    mass = zeros(shape = (q.shape[0], q.shape[1]), dtype='f4')
    raise NotImplemented('fix me')
    print 'num particles rastered', self.mraster(q, ftype, mass, component, quick)
    q[:,:,0]/=mass[:,:]
    q[:,:,1]/=mass[:,:]
    q[:,:,2]/=mass[:,:]
    return X,Y,q.transpose((1,0,2))

  def raster(self, ftype, component, quick=True):
    """ ftype is amongst 'gas', 'star', those with has an sml  """
    field = self.F[ftype]
    result = zeros(dtype='f4', shape = (self.shape[0], self.shape[1]))
    ccode.rasterize(field, result, component, xrange=self.cut[0], yrange=self.cut[1], zrange=self.cut[2], quick=quick)
    return result

  def wraster(self, ftype, component, weightcomponent, quick=True):
    field = self.F[ftype]
    old = field[component].copy()
    try:
      if len(field[component].shape) == 1:
        field[component][:] *= field[weightcomponent][:]
      else:
        field[component][:,0] *= field[weightcomponent][:]
        field[component][:,1] *= field[weightcomponent][:]
        field[component][:,2] *= field[weightcomponent][:]
    
      result = zeros(dtype='f4', shape = (self.shape[0], self.shape[1]))
      mass = zeros(dtype='f4', shape = (self.shape[0], self.shape[1]))
      ccode.rasterize(field, [mass, result], [weightcomponent, component], xrange=self.cut[0], yrange=self.cut[1], zrange=self.cut[2], quick=quick)
      return result, mass
    except Exception as e:
      raise e
    finally:
      field[component] = old

  def linefield(self, ftype, component, src, dir, length=None):
    """ return a line of sight of a field, pos, value """
    f = self.F[ftype]
    tree = f.tree
    if length is None: length = self.boxsize[0]
    pars = tree.trace(src, dir, length)
    pos = f['locations'][pars, :]
    sml = f['sml'][pars]
    mass = f['mass'][pars]
    comp = f[component][pars] * mass
    if component == 'mass':
      Lmass = zeros(shape=1024, dtype='f8')
      ccode.scanline(locations = pos, sml = sml, targets = [Lmass], values = [mass], src=asarray(src), dir=asarray(dir), L=length)
      return linspace(0, length, 1024), Lmass
    else:
      Larray = zeros(shape=1024, dtype='f8')
      Lmass = zeros(shape=1024, dtype='f8')
      ccode.scanline(locations = pos, sml = sml, targets = [Lmass, Larray], values = [mass, comp], src=asarray(src), dir=asarray(dir), L=length)
      Larray /= Lmass
      return linspace(0, length, 1024), Larray

  def camera(self, ftype, component, camera, weightcomponent=None):
#    from gaepsi.tools.simplegl import GLTrans
#    t = GLTrans()
#    t.lookat(target=target, pos=pos, up=up)
#    t.perspective(near=near, far=far, up=up, pos=pos, fov=camera.fov / 360. * 3.1415, aspect=1.0 * raster[0].shape[0] / raster[0].shape[1])

    f = self.F[ftype]
    if weightcomponent is None:
      sph = [float32(f[component])]
      raster = [zeros(dtype='f8', shape = (self.shape[0], self.shape[1]))]
    else:
      m = float32(f[weightcomponent])
      t = float32(f[component] * m)
      sph = [m, t]
      raster = [ zeros(dtype='f8', shape = (self.shape[0], self.shape[1])),
               zeros(dtype='f8', shape = (self.shape[0], self.shape[1])), ]
    if self.periodic: boxsize = asarray(self.boxsize)
    else: boxsize = None
    ccode.camera(raster=raster, sph=sph, locations=f['locations'], 
           sml=f['sml'], near=camera.near, far=camera.far, Fov=camera.fov / 360. * 3.1415, 
           dim=asarray(raster[0].shape), target=asarray(camera.target), up = asarray(camera.up),
           pos=asarray(camera.pos), mask=None, boxsize=boxsize)
    if weightcomponent is None:
      return raster[0], None

    return raster[1], raster[0]

  def projfield(self, ftype, component, weightcomponent=None, use_cache=True):
    """raster a field. ftype can be gas or star. 
       there are two types of components, depending if
       weightcommponent is given.
       If not given, assume component is 'intensive', 
       If given assume component is 'extensive'.

       Extensive component:
       After dividing a particle in two, an extensive quantity is divided by two.
       Examples: mass, sfr.

       Interpolation formula (of component A)
         
         rho_A(x) = sum_i A_i W_i(x, x_i, h_i)

         raster returns the integral of the density rho_A with in a pixel.
         divided by the area of the pixel it becomes the column density of A.

         the pixel integral of rho_A is saved in the cache, if cache enabled.

         returns the pixel mean column density.

       Intensive component:
       After dividing a particle in two, an intensive quantity doesn't change.

       Examples: T, xHI.

       An intensive component has to be weighted by an extensive component.
       For example, weighting the temperature T by total mass or xHI * mass.

       Interpolation formula (of component A, weighted by M)

         A(x) = sum_i A_i M_i / rho_M(x_i) W_i(x, x_i, h_i)
       
         raster is invoked so that it returns an approximation of the
         pixel integral of the M weighted A, as well as the pixel
         integral of M. pixel mean is then obtained by dividing the two.

         returns the pixel weighted integral per pixel, and the integrated weight per pixel.

       """
    if use_cache :
      if ftype not in self.cache1:
        self.cache1[ftype] = {}
      if ftype not in self.cache2:
        self.cache2[ftype] = {}

    if weightcomponent is None:
      if use_cache and component in self.cache1[ftype]:
        return self.cache1[ftype][component] / self.pixel_area, None

      integral = self.raster(ftype, component, quick=False)

      if use_cache:
        self.cache1[ftype][component] = integral.copy()

      integral /= self.pixel_area
      columnden = integral
      return columnden, None

    else:
      cache2name = component + weightcomponent
      if use_cache and cache2name in self.cache2[ftype] and weightcomponent in self.cache1[ftype]:
        return self.cache2[ftype][cache2name].copy() / self.pixel_area, self.cache1[ftype][weightcomponent].copy() / self.pixel_area

      integral, weight_int = self.wraster(ftype, component, weightcomponent, quick=False)

      if use_cache :
        self.cache2[ftype][cache2name] = integral.copy()
        self.cache1[ftype][weightcomponent] = weight_int.copy()

      integral /= self.pixel_area
      weight_int /= self.pixel_area

      return integral, weight_int

  def pointshow(self, ax, ftype, component='bhmass', radius=(4, 1), logscale=True, labelfmt=None, labelcolor='white', vmin=None, vmax=None, count=-1, *args, **kwargs):

    if 'camera' in kwargs:
      return self.bhshow_camera(ax=ax, component=component, radius=radius, logscale=logscale, vmin=vmin, vmax=vmax, count=count, *args, **kwargs)

    from matplotlib.collections import CircleCollection
    mask = self.cut.select(self.F[ftype]['locations'])
    X = self.F[ftype]['locations'][mask,0]
    Y = self.F[ftype]['locations'][mask,1]
    bhmass = self.F[ftype][component][mask]
    if bhmass.size == 0: return

    R = bhmass
    if count > 0: 
      ind = (-R).argsort()
      X = X[ind[0:count]]
      Y = Y[ind[0:count]]
      R = R[ind[0:count]]

    if vmax is None:
      vmax = R.max()
    if vmin is None:
      vmin = ccode.pmin.reduce(R)

    print 'bhshow, vmax, vmin =', vmax, vmin


    if logscale:
      R = log10(R)
      Nm = Normalize(vmax=log10(vmax), vmin=log10(vmin), clip=True)
    else:
      Nm = Normalize(vmax=vmax, vmin=vmin, clip=True)
    if bhmass.size > 1:
      R = Nm(R)

    if R.min() == R.max():
      R = ones(R.shape)

    if isscalar(radius):
      R*=radius
    else:
      rmin = min(radius)
      rmax = max(radius)
      R *= (rmax - rmin)
      R += rmin
    print 'R max, R min', R.min(), R.max()
    print R
#    R.clip(min=4, max=radius**2)
    if not 'marker' in kwargs:
      use_spike = True
    else: 
      use_spike = False
      if not 'edgecolor' in kwargs:
        kwargs['edgecolor'] = 'green'
      if not 'facecolor' in kwargs:
        kwargs['facecolor'] = (0, 1, 0, 0.0)

    if use_spike:
      c = SpikeCollection(X, Y, R, **kwargs)
      ax.add_collection(c)
    else:
      ax.scatter(X,Y, s=R**2, **kwargs)
    if labelfmt: 
      if count > 0:
        ID = ID[ind[0:count]]
      ID = self.F[type]['id'][mask]
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
          color=labelcolor
          )

  def bhshow(self, ax, component='bhmass', radius=(4, 1), logscale=True, labelfmt=None, labelcolor='white', vmin=None, vmax=None, count=-1, *args, **kwargs):
    return self.pointshow(ax, 'bh', component, radius, logscale, labelfmt, labelcolor, vmin, vmax, count, *args, **kwargs)

  #  col = CircleCollection(offsets=zip(X.flat,Y.flat), sizes=(R * radius)**2, edgecolor='green', facecolor='none', transOffset=gca().transData)
  #  ax.add_collection(col)

  def bhshow_camera(self, ax, component='bhmdot', radius=4, logscale=True, vmin=None, vmax=None, count=-1, camera=None, *args, **kwargs):
    bhlum = self.bh.cosmology.QSObol(self.bh[component], 'blue') 

    good = bhlum > 0
    if len(bhlum) > 0 and good.sum() > 0:
      bhlum0 = bhlum[good]
      bhpos0 = self.bh['locations'][good]

    if self.periodic:
      bhpos = zeros((27, bhpos0.shape[0], 3))
      bhlum = zeros((27, bhlum0.shape[0]))

      for im in range(27):
        i, j, k = im / 9 - 1, (im % 9) / 3 -1, im % 3 -1
        bhpos[im, :, :] = bhpos0[:, :] + array([i, j, k]) * self.boxsize
        bhlum[im, :] = bhlum0[:]

      bhpos.shape = -1, 3
      bhlum.shape = -1
    else:
      bhpos = bhpos0
      bhlum = bhlum0

    bhpos = camera.map(bhpos, self.shape)
    bhdist = bhpos[:, 2]
    bhlum /= 4 * 3.1416 * bhdist ** 2
    bhlum /= (self.bh.cosmology.units.W / self.bh.cosmology.units.METER ** 2 )

    print 'distance of bhs', bhdist.min(), bhdist.max()
    print 'lum of bhs', bhlum.min(), bhlum.max()

# apparent maginitute
    mag = ((-26.74 + 2.5 * log10(1400 / bhlum)))
    ind = mag.argsort()
    if count > -1 and count < len(mag):
      magcut = mag[ind[count]]
    else:
      magcut = 9999
# 8 is the naked eye limit 
    selected = (bhpos[:, 0] > 0) & (bhpos[:, 0] < self.shape[0]) 
    selected &= (bhpos[:, 1] > 0) & (bhpos[:, 1] < self.shape[1]) 
    selected &= (bhpos[:, 2] > 0.0)& (mag  < magcut)
    print 'mag', mag.min(), mag.max()
    bhpos = bhpos[selected]
    bhlum = bhlum[selected]
    print 'bhs shown', selected.sum()
    mag = - mag[selected]
    mag -= mag.min()
    diff = mag.max()
    if diff > 0: 
      mag /= diff
    else: mag[:] = 1
     
    bhx = bhpos[:, 0]
    bhy = bhpos[:, 1]
    print mag
    t = 0.05
    marker = ((-1, 0), (-t, t), (0, 1), (t, t), (1, 0), (t, -t), (0, -1), (-t, -t)), 0
    color=(0.8, 1, 1, 0.7)
    ecolor=(1., 1, 0.8, 0.2)

    if isscalar(radius):
      mag*=radius
    else:
      rmin = min(radius)
      rmax = max(radius)
      mag *= (rmax - rmin)
      mag += rmin
    c = SpikeCollection(bhx, bhy, radius=mag, color=color)
    ax.add_collection(c)
#    ax.scatter(x=bhx, y=bhy, s=mag * 10**2, marker=marker, edgecolor=ecolor, color=color)

  def velshow(self, ax, ftype, relative=False, color='cyan', alpha=0.8):
    X,Y,vel = self.vector(ftype, 'vel', grids=(20,20), quick=False)
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
    ax.quiver(X,Y, vel[:,:,0], vel[:,:,1], width=0.003, scale=scale, scale_units='width', angles='xy', color=color,alpha=alpha)
    if mean != None: 
      ax.quiver(self.cut.center[0], self.cut.center[1], mean[0], mean[1], scale=scale, scale_units='width', width=0.01, angles='xy', color=(0,0,0,0.5))


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

  def reset_view(self, ax, camera=False):
    if camera:
      left, right = 0, self.shape[0]
      bottom, top = 0, self.shape[1]
    else:
      left,right =self.cut['x']
      bottom, top = self.cut['y']
    print left, right, bottom, top
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)

  def makeT(self, Xh = 0.76):
    """T will be in Kelvin"""
    gas =self.gas
    C = gas.cosmology
    gas['T'] = zeros(dtype='f4', shape=gas.numpoints)
    C.ie2T(ie = gas['ie'], ye = gas['ye'], Xh = Xh, out = gas['T'])
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
    for comp in bh.names: bh[comp] = bh[comp][ind]

  def fieldshow(self, ax, ftype, component, mode=None, camera=None,
    vmin=None, vmax=None,
    logscale=True, cmap=pygascmap,
    gamma=1.0, mmin=None, mmax=None, over=None,
    return_raster=False, logweight=None, weightcomponent='mass', use_cache=True):

    if camera is None:
      if mode is None: # extensive, mass->density, sfr->sf density, 
        todraw, mass = self.projfield(ftype=ftype, component=component, use_cache=use_cache)
      else : # intensive, decorate the result aftwards
        todraw, mass = self.projfield(ftype=ftype, component=component, weightcomponent=weightcomponent, use_cache=use_cache)
    else:
      if mode is None: # extensive, mass->density, sfr->sf density, 
        todraw, mass = self.camera(ftype=ftype, component=component, camera=camera)
      else : # intensive, decorate the result aftwards
        todraw, mass = self.camera(ftype=ftype, component=component, weightcomponent=weightcomponent, camera=camera)

    if camera is None:
      if logweight is None: logweight=True
      image = self.render(todraw=todraw, mass=mass, mode=mode, vmin=vmin, vmax=vmax, logscale=logscale, logweight=logweight, cmap=cmap, gamma=gamma, mmin=mmin, mmax=mmax, over=over)
      ret = ax.imshow(image.transpose((1,0,2)), origin='lower',
         extent=self.extent, vmin=vmin, vmax=vmax, cmap=cmap)
    else:
      if logweight is None: logweight=False
      image = self.render(todraw=todraw, mass=mass, mode=mode, vmin=vmin, vmax=vmax, logscale=logscale, logweight=logweight, cmap=cmap, gamma=gamma, mmin=mmin, mmax=mmax, over=over)
      ret = ax.imshow(image.transpose((1,0,2)), origin='lower',
         extent=(0, self.shape[0], 0, self.shape[1]), vmin=vmin, vmax=vmax, cmap=cmap)
    return ret

  def fieldcontour(self, ax, ftype, component, camera=None,
    levels=None, logweight=None, weightcomponent='mass', use_cache=True, colors='k', linewidth=2, linestyle=['solid']):

    if camera is None:
      todraw, mass = self.projfield(ftype=ftype, component=component, weightcomponent=weightcomponent, use_cache=use_cache)
    else:
      todraw, mass = self.camera(ftype=ftype, component=component, weightcomponent=weightcomponent, camera=camera)

    if camera is None:
      if logweight is None: logweight=True
      ret = ax.contourf((todraw/mass).T, extent=self.extent, colors=colors, linewidth=linewidth, levels=levels, linestyle=linestyle)
    else:
      if logweight is None: logweight=False
      ret = ax.contourf((todraw/mass).T, extent=(0, self.shape[0], 0, self.shape[1]), colors=colors, linewidth=linewidth, levels=levels, linestyle=linestyle)
  
    return ret
     
  def render(self, todraw, mass, mode=None, vmin=None, vmax=None, logscale=True, logweight=True, cmap=pygascmap, gamma=1.0, mmin=None, mmax=None, over=None, composite=None):
    """ vmin can be a string, eg '40 dB', so the vmin=vmax/1e4 """
    if mode == 'intensity' or mode == 'mean':
      todraw /= mass

    if logscale:
     if vmin is not None and not isinstance(vmin, basestring):
       vmin = 10 ** vmin
     if vmax is not None:
       vmax = 10 ** vmax

    if vmax is None: 
      if mode == 'intensity' or mode == 'mean' or over is None:
        vmax = fmax.reduce(todraw.flat)
      else:
        sort = todraw.ravel().argsort()
        ind = sort[(len(sort) - 1) * (1.0 - over)]
        vmax = todraw.ravel()[ind]
        del sort
    if vmin is None: 
      if logscale:
        vmin = ccode.pmin.reduce(todraw.flat)
      else:
        vmin = fmin.reduce(todraw.flat)
    elif isinstance(vmin, basestring):
      db = int(vmin[:-3])
      vmin = vmax / 10 ** (db/10.)
    if logscale:
      log10(todraw, todraw)
      vmin = log10(vmin)
      vmax = log10(vmax)

    print 'vmax, vmin =' , vmax, vmin

    image = zeros(dtype = ('u1', 4), shape = todraw.shape)

    # gacolor is much faster then matplotlib's normalize and uses less memory(4times fewer).
    if not hasattr(cmap, 'table'):
      cmap = gacmap(cmap)

    ccode.color(image, todraw, max = vmax, min = vmin, logscale=False, colormap = cmap)
    del todraw
    if mode == 'intensity':
      weight = mass
      if logweight:
      # in camera mode do not use logscale for mass.
        weight.clip(ccode.pmin.reduce(weight.ravel()), inf, weight)
        log10(weight, weight)
      print 'weight', weight.mean(), weight.max(), weight.min()
      if mmin is None:
        mmin = weight.min()
      if mmax is None:
        if over is None:
          mmax = weight.max()
        else:
          sort = weight.ravel().argsort()
          ind = sort[(len(sort) - 1) * (1.0 - over)]
          mmax = weight.ravel()[ind]
      print 'mmax, mmin', mmax, mmin
      weight -= mmin
      weight /= (mmax - mmin)
      weight.clip(0, 1, weight)
    else:
      weight = image[:, :, 3] / 255.0

    weight **= gamma

    if composite is not None:
      alpha = composite[:, :, 3] / 255.0
      multiply(1.0 - weight[:, :], alpha[:, :], alpha[:, :])
      multiply(image[:, :, 0:3], weight[:, :, newaxis], image[:, :, 0:3])
      multiply(composite[:, :, 0:3], alpha[:, :, newaxis], composite[:, :, 0:3])
      add(image[:, :, 0:3], composite[:, :, 0:3], image[:, :, 0:3])
      image[:, :, 3] = (alpha[:, :] + weight[:, :]) * 255
    else:
      #multiply(image[:, :, 0:3], weight[:, :, newaxis], image[:, :, 0:3])
      #image[:, :, 3] = 255
      #weight **= 0.33333333333
      multiply(255.9999, weight[:, :], out=image[:,:,3], casting='unsafe')
#      print 'alpha', image[:, :, 3].ravel().min(), image[:,:,3].ravel().max()
    return image

  def decorate(self, ax, frameon=True, title=None, bgcolor='k', color='w', fontsize='small'):
    cut = self.cut
    ax.set_axis_bgcolor(bgcolor)
    if frameon :
      ax.ticklabel_format(axis='x', useOffset=cut.center[0])
      ax.ticklabel_format(axis='y', useOffset=cut.center[1])
      ax.set_xticks(linspace(cut['x'][0], cut['x'][1], 5))
      ax.set_yticks(linspace(cut['y'][0], cut['y'][1], 5))
      #ax.set_title(title)
      if(title is not None):
        ax.set_title(title, position=(0.1, 0.9), fontsize=fontsize, color=color, transform=ax.transAxes)
    else :
      ax.axison = False
      if(title is not None):
        ax.set_title(title, position=(0.1, 0.9), fontsize=fontsize, color=color, transform=ax.transAxes)
      ax.figure.set_facecolor(bgcolor)

  def drawscale(self, ax, color='white', fontsize=None):
    from mpl_toolkits.axes_grid.anchored_artists import AnchoredSizeBar
    l = (self.cut.size[0]) * 0.2
    l = l // 10 ** int(log10(l)) * 10 ** int(log10(l))
    if l > 500 :
      l/=1000.0
      l = int(l+0.5)
      text = r"%g Mpc/h" % l
      l *= 1000.0
    else:
      text = r"%g Kpc/h" %l
   
    print text
    b = AnchoredSizeBar(ax.transData, l, text, loc = 8, 
        pad=0.1, borderpad=0.5, sep=5, frameon=False)
    for r in b.size_bar.findobj(Rectangle):
      r.set_edgecolor(color)
    for t in b.txt_label.findobj(Text):
      t.set_color(color)
      if fontsize is not None:
        t.set_fontsize(fontsize)

    ax.add_artist(b)

  def circle(self, ax, *args, **kwargs):
    from matplotlib.patches import Circle
    c = Circle(*args, **kwargs)
    ax.add_patch(c)

context = GaplotContext()

def __ensure__(figure):
  if not has_attr(figure, 'gaplot'):
    figure.gaplot = GaplotContext()

def read(*args, **kwargs):
  return context.read(*args, **kwargs)
read.__doc__ = GaplotContext.read.__doc__

def use(*args, **kwargs):
  return context.use(*args, **kwargs)
use.__doc__ = GaplotContext.use.__doc__

def zoom(*args, **kwargs):
  return context.zoom(*args, **kwargs)
zoom.__doc__ = GaplotContext.zoom.__doc__

def autozoom(*args, **kwargs):
  return context.autozoom(*args, **kwargs)
autozoom.__doc__ = GaplotContext.autozoom.__doc__

def slice(*args, **kwargs):
  return context.slice(*args, **kwargs)
zoom.__doc__ = GaplotContext.slice.__doc__

def unfold(*args, **kwargs):
  return context.unfold(*args, **kwargs)

def rotate(*args, **kwargs):
  return context.rotate(*args, **kwargs)

def circle(ax=None, *args, **kwargs):
  if ax is None: ax = gca()
  context.circle(ax, *args, **kwargs)
  draw()
circle.__doc__ = GaplotContext.circle.__doc__

def bhshow(ax=None, *args, **kwargs):
  if ax is None: ax = gca()
  context.bhshow(ax, *args, **kwargs)
  draw()
bhshow.__doc__ = GaplotContext.bhshow.__doc__

def reset_view(ax=None, camera=False):
  if ax is None: ax = gca()
  context.reset_view(ax, camera)
  draw()

def drawscale(ax=None, *args, **kwargs):
  if ax is None: ax = gca()
  context.drawscale(ax, *args, **kwargs)
  draw()

def gasshow(component='mass', ax=None, *args, **kwargs):
  "see fieldshow for docs"
  kwargs['component'] = component
  ftype = kwargs.pop('ftype', 'gas')
  if ax is None: ax = gca()
  ret = context.fieldshow(ax, ftype, *args, **kwargs)
  ax._sci(ret)
  draw()

def fieldcontour(ftype='gas', component='mass', use_figimage=False, ax=None, *args, **kwargs):
  "see fieldshow for docs"
  kwargs['component'] = component
  if ax is None: ax = gca()
  ret = context.fieldcontour(ax, ftype, *args, **kwargs)
  ax._sci(ret)
  draw()

def starshow(component='mass', ax=None, *args, **kwargs):
  kwargs['ftype'] = 'star'
  gasshow(component, ax, *args, **kwargs)

def starshow_poor(ax=None, *args, **kwargs):
  if ax is None: ax = gca()
  context.starshow_poor(ax, *args, **kwargs)
  draw()

def decorate(ax=None, *args, **kwargs):
  if ax is None: ax = gca()
  context.decorate(ax=ax, *args, **kwargs)
  draw()

def velshow(ftype, ax=None, relative=False, color='cyan', alpha=0.8):
  if ax is None: ax = gca()
  context.velshow()
  draw()
