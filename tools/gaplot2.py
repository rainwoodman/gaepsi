#! python
import numpy 
import matplotlib.pyplot as pyplot
from gaepsi.snapshot import Snapshot
from gaepsi.field import Field, Cut
from gaepsi.tools.keyframes import KeyFrames
from gaepsi import ccode
from gaepsi.cython import _camera
from gaepsi.tools.meshmap import Meshmap
from gaepsi.tools.spikes import SpikeCollection
from gaepsi.tools import sharedmem

def _ensurelist(a):
  if numpy.isscalar(a):
    return (a,)
  return a

def normalize(value, vmin=None, vmax=None, logscale=False):
    "normalize an array inplace"
    if logscale:
      numpy.log10(value, value)
    if vmax is None:
      vmax = numpy.nanmax(value)
    if vmin is None:
      vmin = numpy.nanmin(value)
    elif isinstance(vmin, basestring):
      if logscale and 'db' in vmin:
        vmin = vmax - float(vmin[:-2]) * 0.1 
      else:
        raise ValueError('vmin format is "?? db"')

    value[...] -= vmin
    value[...] /= (vmax - vmin)
    value.clip(0, 1, out=value)

def image(color, cmap, luminosity=None, composite=False):
    image = cmap(color, bytes=True)
    if luminosity is not None:
      luminosity = luminosity.copy()
      if composite:
        numpy.multiply(luminosity, 255.999, image[..., :3], casting='unsafe')
      else:
        numpy.multiply(image[...,:3], luminosity[..., None], image[..., :3], casting='unsafe')

    return image


class GaplotContext(object):
  def __init__(self, shape = (600,600)):
    self.format = None
    self.shape = shape

    # fields
    self.F = {}
    # ptypes
    self.P = {}
    # Trees
    self.T = {}
    # VT cache
    self.VT = {}

    self.cut = Cut()
    self.periodic = False
    self.size = numpy.array([1, 1, 1.])
    self.center = numpy.array([0, 0, 0.])
    self.up = numpy.array([0, 1, 0.])
    self.dir = numpy.array([0, 0, 1.])

  def invalidate(self):
    pass

  @property
  def extent(self):
    l = -self.size[0] * 0.5 + self.center[0]
    r = self.size[0] * 0.5 + self.center[0]
    b = -self.size[1] * 0.5 + self.center[1]
    t = self.size[1] * 0.5 + self.center[1]
    return numpy.array([l, r, b , t])

  def view(self, center=None, size=None, up=[0,1,0], dir=[0,0,-1]):
    if center == 'auto':
      for ftype in self.F:
        if self.F[ftype].numpoints > 0:
          min=self.F[ftype]['locations'].min(axis=0)
          max=self.F[ftype]['locations'].max(axis=0)
          center = (min + max) * 0.5
          size = (max - min) 
          break

    self.center= numpy.ones(3) * center
    self.size= numpy.ones(3) * size
    self.up = numpy.ones(3) * up
    self.dir = numpy.ones(3) * dir

  def _mkcamera(self, cameraClass):
    if cameraClass is None: 
      cameraClass = _camera.OCamera

    camera = cameraClass(width=self.shape[0], height=self.shape[1])
    camera.lookat(target=self.center,
       pos=self.center-self.dir * self.size[2], up=self.up)
    if isinstance(camera, _camera.OCamera):
      camera.zoom(extent=(-self.size[0] * 0.5, self.size[0] * 0.5, -self.size[1] * 0.5, self.size[1] * 0.5), near=0.5 * self.size[2], far=1.5*self.size[2])
    elif isinstance(camera, _camera.PCamera):
      fov = numpy.arctan2(self.size[0], self.size[2]) * 0.5
      aspect = self.size[1] / self.size[0]
      camera.zoom(fov=fov, aspect=aspect, near=0.5 * self.size[2], far=1.5*self.size[2])
    return camera

  def paint(self, ftype, color, luminosity, sml=None, cameraClass=None, fade=True):
    camera = self._mkcamera(cameraClass)
    camera.fade = fade
    if color is not None:
      acolor = self.F[ftype][color]
    else:
      acolor = None
    if luminosity is not None:
      aluminosity = self.F[ftype][luminosity]
    else:
      aluminosity = None
    CCD = numpy.zeros(self.shape, dtype=('f8',2))
    if sml is None:
      if not (ftype, color, luminosity) in self.VT:
        self.VT[(ftype, color, luminosity)] = _camera.VisTree(self.T[ftype], acolor, aluminosity)
      vt = self.VT[(ftype, color, luminosity)]
      nodes = vt.find_large_nodes(camera, 0, 65536*8)
      with sharedmem.Pool(use_threads=True) as pool:
        def work(node):
          _CCD = vt.paint(camera, node)
          with pool.lock:
            CCD[...] += _CCD
        pool.map(work, nodes)
    else:
      sml = self.F[ftype][sml]
      x,y,z=self.F[ftype]['locations'].T
      with sharedmem.Pool(use_threads=True) as pool:
        def work(x,y,z,sml,color,luminosity):
          _CCD = camera.paint(x,y,z,sml,color,luminosity)
          with pool.lock:
            CCD[...] += _CCD
        pool.starmap(work, pool.zipsplit((x,y,z,sml,acolor,aluminosity)))
    C, L = CCD[...,0], CCD[...,1]
    return C/L, L
    
  def transform(self, ftype, luminosity=None, bleeding=0, cameraClass=None, fade=True):
    """ returns the CCD coordinate of inside objects 
        and a mask, True if object is visible on the CCD.
        x, y, mask.
        The actually range used is 
        [-bleeding, width+bleeding] x [-bledding, height+bleeding]
        if luminosity is given, 
          it is faded with square invesre law if fade is True
    """
    camera = self._mkcamera(cameraClass)
    pos = self.F[ftype]['locations']
    x,y,z = pos[:, 0], pos[:, 1], pos[:, 2]
    uvt = camera.transform(x, y, z)
    x = (uvt[:, 0] + 1.0) * 0.5 * self.shape[0]
    y = (uvt[:, 1] + 1.0) * 0.5 * self.shape[1]
    mask = x >= -bleeding
    mask&= y >= -bleeding
    mask&= x <= self.shape[0]+bleeding
    mask&= y <= self.shape[1]+bleeding
    mask&= uvt[:,2] > -1
    mask&= uvt[:,2] < 1
    D = ((pos - self.center)**2).sum(axis=-1)
    if luminosity is not None:
      if fade:
        return x, y, mask, luminosity /(3.1416*D)
      else:
        return x, y, mask, luminosity.copy()
    if luminosity is None:
      return x, y, mask
    


  def imshow(self, ax, image):
    ax.imshow(image.swapaxes(0,1), origin='lower', extent=self.extent)

  def scatter(self, ax, x, y, mask, s):
    x = x[mask]
    y = y[mask]
    l, r, b, t =self.extent
    x = x / self.shape[0] * (r - l) + l
    y = y / self.shape[1] * (t - b) + b
    ax.scatter(x, y, s[mask]) 
    
  def schema(self, ftype, types, components):
    self.F[ftype] = Field(components=components)
    self.P[ftype] = _ensurelist(types)

  def use(self, snapname, format, periodic=False, cut=None):
    self.snapname = snapname
    self.format = format
    try:
      snapname = self.snapname % 0
    except TypeError:
      snapname = self.snapname

    snap = Snapshot(snapname, self.format)

    for ftype in self.F:
      self.F[ftype].init_from_snapshot(snap)

    self.C = snap.C
    if cut is not None:
      self.cut.take(cut)
    else:
      try:
        boxsize = snap.C['boxsize']
        self.cut.take(Cut(xcut=[0, boxsize], ycut=[0, boxsize], zcut=[0, boxsize]))
      except:
        pass

    self.boxsize = numpy.ones(3) * snap.C['boxsize']
    self.redshift = snap.C['redshift']
    self.periodic = periodic
    self.invalidate()
 
  def read(self, ftypes, fids=None, numthreads=None):
    if fids is not None:
      snapnames = [self.snapname % i for i in fids]
    elif '%d' in self.snapname:
      snapnames = [self.snapname % i for i in range(self.C['Nfiles'])]
    else:
      snapnames = [self.snapname]
    snapshots = [Snapshot(snapname, self.format) for snapname in snapnames]

    for ftype in _ensurelist(ftypes):
      self.F[ftype].take_snapshots(snapshots, ptype=self.P[ftype], nthreads=numthreads, cut=self.cut)
      self.T[ftype] = self.F[ftype].zorder(ztree=True, thresh=32)
    self.invalidate()

  def reset_view(self, ax, camera=False):
    l,r,b,t = self.extent
    ax.set_xlim(l, r)
    ax.set_ylim(b, t)

  def frame(self, ax, off=False, bgcolor='k'):
    if not off:
      ax.set_axis_bgcolor(bgcolor)
      l,r,b,t = self.extent
      ax.ticklabel_format(axis='x', useOffset=self.center[0])
      ax.ticklabel_format(axis='y', useOffset=self.center[1])
      ax.set_xticks(numpy.linspace(l, r, 5))
      ax.set_yticks(numpy.linspace(b, t, 5))
    else :
      ax.axison = False
      ax.figure.set_facecolor(color)

  def drawscale(self, ax, color='white', fontsize=None):
    from mpl_toolkits.axes_grid.anchored_artists import AnchoredSizeBar
    from matplotlib.patches import Rectangle
    from matplotlib.text import Text

    l = (self.size[0]) * 0.2
    l = l // 10 ** int(numpy.log10(l)) * 10 ** int(numpy.log10(l))
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
      r.set_edgecolor(color)
    for t in b.txt_label.findobj(Text):
      t.set_color(color)
      if fontsize is not None:
        t.set_fontsize(fontsize)

    ax.add_artist(b)

  def makeT(self, ftype, Xh = 0.76):
    """T will be in Kelvin"""
    gas =self.F[ftype]
    C = gas.cosmology
    gas['T'] = numpy.zeros(dtype='f4', shape=gas.numpoints)
    C.ie2T(ie = gas['ie'], ye = gas['ye'], Xh = Xh, out = gas['T'])
    gas['T'] *= C.units.TEMPERATURE

context = GaplotContext()

for x in dir(context):
  from functools import update_wrapper
  import inspect
  m = getattr(context, x)
  if not x.startswith('_') and hasattr(m, '__call__'):
    locals()[x] = m

if False:
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

