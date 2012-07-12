#! python
import numpy 
import matplotlib.pyplot as pyplot
from gaepsi.snapshot import Snapshot
from gaepsi.field import Field, Cut
from gaepsi.readers import Reader
from gaepsi.cython import _camera
from gaepsi.cython import _fast
from gaepsi.tools.meshmap import Meshmap
from gaepsi.tools.spikes import SpikeCollection
import sharedmem
from gaepsi.cosmology import Cosmology

def _ensurelist(a):
  if numpy.isscalar(a):
    return (a,)
  return a

def _fr10(n):
  """ base 10 frexp """
  exp = numpy.floor(numpy.log10(n))
  return n * 10 ** -exp, exp

def n_(value, vmin=None, vmax=None):
  return normalize(value, vmin, vmax, logscale=False)

def nl_(value, vmin=None, vmax=None):
  return normalize(value, vmin, vmax, logscale=True)

class normalize(numpy.ndarray):
  def __new__(cls, value, vmin=None, vmax=None, logscale=False, out=None):
    """normalize an array to 0 and 1, value is returned.
       if logscale is True, and vmin is in format 'nn db', then
       vmin = vmax - nn * 0.1
       NaNs and INFs are neglected
       safe for in-place operation
       returns normalized_value, vmin, vmax
         where vmin, vmax are the real vmin vmax used
    """
    if out is None:
      out = value.copy()

    if logscale:
      numpy.log10(value, out)
    if vmax is None:
      vmax = _fast.finitemax(out)
    elif isinstance(vmax, basestring) \
        and '%' == vmax[:-1]:
      vmax = numpy.percentile(out, float(vmax[:-1]))

    if vmin is None:
      vmin = _fast.finitemin(out)
    elif isinstance(vmin, basestring):
      if logscale and 'db' in vmin:
        vmin = vmax - float(vmin[:-2]) * 0.1 
      else:
        raise ValueError('vmin format is "?? db"')

    out[...] -= vmin
    out[...] /= (vmax - vmin)
    out.clip(0, 1, out=out)
    self = out.view(type=normalize)
    self.vmin = vmin
    self.vmax = vmax
    self.logscale = logscale
    return self

  def __repr__(self):
    return 'normed values = %s, vmin=%s, vmax=%s, logscale=%s' % \
          (repr(self.view(type=numpy.ndarray)), 
           repr(self.vmin), repr(self.vmax), repr(self.logscale))

def image(color, cmap, luminosity=None, composite=False):
    """converts (color, luminosity) to an rgba image,
       if composite is False, directly reduce luminosity
       on the RGBA channel by luminosity,
       if composite is True, reduce the alpha channel.
       color and luminosity needs to be normalized
    """
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
    # fields
    self.F = {}
    # ptypes
    self.P = {}
    # Trees
    self.T = {}
    # VT cache
    self.VT = {}

    self.periodic = False
    self.origin = numpy.array([0, 0, 0.])
    self.boxsize = numpy.array([1, 1, 1.])

    self._shape = shape
    self.view(center=[0, 0, 0], size=[1, 1, 1], up=[0, 1, 0], dir=[0, 0, -1], fade=True, method='ortho')
    from gaepsi.cosmology import default
    self.cosmology = default
    self.units = self.cosmology.units
    
    # used to save the last state of plotting routines
    self.last = {}

  @property
  def shape(self):
    return self._shape
  @shape.setter
  def shape(self, shape):
    self._shape = (shape[0], shape[1])
    self.view()
    
  def __getitem__(self, ftype):
    return self.F[ftype]

  def __setitem__(self, ftype, value):
    if not isinstance(value, Field):
      raise TypeError('need a Field')
    self.F[ftype] = value
    self.rebuildtree(ftype)

  def rebuildtree(self, ftype, thresh=32):
    self.T[ftype] = self.F[ftype].zorder(ztree=True, thresh=thresh)
    if ftype in self.VT:
      del self.VT[ftype]

  def invalidate(self):
    pass

  @property
  def extent(self):
    l = -self.size[0] * 0.5
    r = self.size[0] * 0.5
    b = -self.size[1] * 0.5
    t = self.size[1] * 0.5
    return l, r, b , t

  def view(self, center=None, size=None, up=None, dir=None, method=None, fade=None):
    """ 
       if center is a field type, 
         center will be calcualted from that field type
         if size is None, size will be derived, too

       in all othercases, anything non-none will be overwritten,

       A new default camera is always made:w
    """
    if isinstance(center, basestring):
      ftype = center
      if self.F[ftype].numpoints > 0:
        t = self.T[ftype].digitize
        center =t.min + 0.5 * t.scale
      if size is None:
        size= t.scale.copy()

    if center is not None:
      self.center = numpy.ones(3) * center
    if size is not None:
      self.size = numpy.ones(3) * size
    if up is not None:
      self.up = numpy.ones(3) * up
    if dir is not None:
      self.dir = numpy.ones(3) * dir
    if method is not None:
      self.method = method
    if fade is not None:
      self.fade = fade

    self.default_camera = self._mkcamera(self.method, self.fade)

  def _mkcamera(self, method, fade):
    """ make a camera in the given class,
        for PCamera, assuming the extent is
        the field of view at the target distance,
        for OCamera, directly set the field of view
        also set the fade property of the camera,
        determining whether the distance is used
        to simulate the brightness reduction.
    """

    target = self.center
    distance = self.size[2]

    camera = _camera.Camera(width=self.shape[0], height=self.shape[1])
    camera.lookat(target=target,
       pos=target - self.dir * distance, up=self.up)
    if method == 'ortho':
      camera.ortho(extent=(-self.size[0] * 0.5, self.size[0] * 0.5, -self.size[1] * 0.5, self.size[1] * 0.5), near=distance - 0.5 * self.size[2], far=distance + 0.5 *self.size[2])
    elif method == 'persp':
      fov = numpy.arctan2(self.size[1] * 0.5, distance) * 2
      aspect = self.size[0] / self.size[1]
      camera.persp(fov=fov, aspect=aspect, near=distance - 0.5 * self.size[2], far=distance + 0.5*self.size[2])
    camera.fade = fade
    return camera

  def _mkcameras(self, camera=None):
    if not camera:
      camera = self.default_camera
    if not self.periodic:
      yield camera
      return
    target_residual = numpy.remainder(camera.target - self.origin, self.boxsize)
    shift = target_residual - camera.target
    pos_residual = camera.pos + shift
    
    x, y, z = 0.5 * self.boxsize
    for celloffset in numpy.broadcast(*numpy.ogrid[-2:3,-2:3,-2:3]):
      c = camera.copy()
      c.lookat(pos=pos_residual+self.boxsize * celloffset,
               target=target_residual+self.boxsize * celloffset)
      if c.dir.dot(camera.dir) < 0:
        continue
      if c.mask(x, y, z, (x,y,z)):
        yield c
      else:
        continue


  def paint(self, ftype, color, luminosity, sml=None, camera=None):
    """ paint field to CCD, returns
        C, L where
          C is the color of the pixel
          L is the exposure of the pixel
        Notice that if color is None, 
          C will be undefined,
          L will still be the exposure.
        the return values can be normalized by
        nl_ or n_, then feed to imshow
    """
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
      if not (ftype, color, luminosity) in self.VT \
      or self.VT[(ftype, color, luminosity)].tree != self.T[ftype]:
        self.VT[(ftype, color, luminosity)] = _camera.VisTree(self.T[ftype], acolor, aluminosity)
      vt = self.VT[(ftype, color, luminosity)]
      for cam in self._mkcameras(camera):
        print 'cam:', cam.pos, cam.dir, cam.up
        nodes = vt.find_large_nodes(cam, 0, 65536*8)
        with sharedmem.Pool(use_threads=True) as pool:
          def work(node):
            _CCD = vt.paint(cam, node)
            with pool.lock:
              CCD[...] += _CCD
          pool.map(work, nodes)
    else:
      if sml in self.F[ftype]:
        sml = self.F[ftype][sml]
      print 'sml', sml
      x,y,z=self.F[ftype]['locations'].T
      for cam in self._mkcameras(camera):
        print 'cam:', cam.pos, cam.dir, cam.up
        with sharedmem.Pool(use_threads=True) as pool:
          def work(x,y,z,sml,color,luminosity):
            _CCD = cam.paint(x,y,z,sml,color,luminosity)
            with pool.lock:
              CCD[...] += _CCD
          pool.starmap(work, pool.zipsplit((x,y,z,sml,acolor,aluminosity)))
    C, L = CCD[...,0], CCD[...,1]
    return C/L, L
    
  def transform(self, ftype, luminosity=None, radius=0, camera=None):
    """ Find the CCD coordinate of objects inside the field of view. Objects are of 0 size, 
        r is the bounding radius of radius of the plotted object.
        The actually range used is 
        [-r, width+radius] x [-bledding, height+radius]
        If luminosity is given
          returns X, Y, B, where B is the apparent brightness according to the Camera [ depending fade is True or not ]
        otherwise
          returns X, Y, I where I is the index of object in the original field. One object may show up multiple times because of the periodic boundary condition.
          
        
    """
    X, Y, D, L, I = [], [], [], [], []
    if luminosity is not None:
      alum = self.F[ftype][luminosity]
    else:
      alum = None
    for cam in self._mkcameras(camera):
      pos = self.F[ftype]['locations']
      x,y,z = pos[:, 0], pos[:, 1], pos[:, 2]
      uvt = cam.transform(x, y, z)
      x = (uvt[:, 0] + 1.0) * 0.5 * self.shape[0]
      y = (uvt[:, 1] + 1.0) * 0.5 * self.shape[1]
      mask = x >= -radius
      mask&= y >= -radius
      mask&= x <= self.shape[0]+radius
      mask&= y <= self.shape[1]+radius
      mask&= uvt[:,2] > -1
      mask&= uvt[:,2] < 1
      X += [x[mask]]
      Y += [y[mask]]
      if alum is not None:
        d = ((pos - cam.pos)**2).sum(axis=-1)
        D += [d[mask]]
        L += [alum[mask]]
      else:
        I += [mask.nonzero()[0]]
    X = numpy.concatenate(X)
    Y = numpy.concatenate(Y)
    if alum is not None:
      D = numpy.concatenate(D)
      L = numpy.concatenate(L)
      if cam.fade:
        return X, Y, L /(3.1416*D)
      else:
        return X, Y, L
    else:
      I = numpy.concatenate(I)
      return X, Y, I

  def colorbar(self, ax=None, **kwargs):
    from matplotlib import colorbar as cb
    from matplotlib.ticker import Formatter
    class MyFormatter(Formatter):
      def __init__(self, logscale, vmin, vmax):
        self.logscale = logscale
        self.vmin = vmin
        self.vmax = vmax
        self.scale = 10 **(_fr10(vmax - vmin)[1] - 1)
        if vmax != 0 and \
           numpy.abs((vmin - vmax) / vmax) < 0.01:
          self.offset = vmax
        else:
          self.offset = 0
      def get_offset(self):
        if self.offset != 0:
          return '+%.3g\n x%.3g' % (self.offset, self.scale)
        else:
          return r'x%.3g' % self.scale
      def __call__(self, data, pos=None):
        if self.offset != 0:
          return '%.3g' % ((data - self.offset) / self.scale)
        else:
          return '%.3g' % (data / self.scale)

    if not hasattr(ax, 'colorbarax'):
      ca = ax.get_figure().gca()
      ax.colorbarax, kwargs = cb.make_axes(ax, **kwargs)
      ax.get_figure().sca(ca)
    color = context.last['color']
    cb.ColorbarBase(ax=ax.colorbarax, 
           cmap=self.last['cmap'], 
           norm=cb.colors.Normalize(
             vmin=color.vmin, vmax=color.vmax),
           format = MyFormatter(color.logscale, color.vmin, color.vmax)
           )

  def imshow(self, color, luminosity=None, ax=None, **kwargs):
    """ shows an image on ax.
        always expecting color and luminosity
        normalized to [0, 1].
        
        The default color map is 
          coolwarm if luminosity is given
          gist_heat if luminosity is not given
        Example:

        >> C, L = paint('ie', 'mass')
        >> imshow(nl_(C), nl_(L))
        
    """
    realkwargs = dict(origin='lower', extent=self.extent)
    if luminosity is not None:
       realkwargs['cmap'] = pyplot.cm.coolwarm
    else:
       realkwargs['cmap'] = pyplot.cm.gist_heat
    realkwargs.update(kwargs)
    self.last['cmap'] = realkwargs['cmap']
    cmap = realkwargs['cmap']
    if not isinstance(color, normalize):
      color = nl_(color)
    if luminosity is not None and not isinstance(luminosity, normalize):
      luminosity = nl_(luminosity)
      
    im = image(color, cmap, luminosity)
    ax.imshow(im.swapaxes(0,1), **realkwargs)

    self.last['color'] = color

  def scatter(self, x, y, s, ax=None, **kwargs):
    x = x
    y = y
    l, r, b, t =self.extent
    x = x / self.shape[0] * (r - l) + l
    y = y / self.shape[1] * (t - b) + b
    ax.scatter(x, y, s, **kwargs)
    
  def schema(self, ftype, types, components):
    """ loc dtype is the base dtype of the locations."""
    reader = Reader(self.format)
    schemed = {}
    for comp in components:
      if comp is tuple:
        schemed[comp[0]] = comp[1]
      elif comp in reader:
        schemed[comp] = reader[comp].dtype

    if 'pos' in reader:
      self.F[ftype] = Field(components=schemed, dtype=reader['pos'].dtype.base)
    else:
      self.F[ftype] = Field(components=schemed, dtype=None)

    self.P[ftype] = _ensurelist(types)

  def use(self, snapname, format, periodic=False, origin=[0,0,0.], boxsize=None):
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
    self.origin[...] = numpy.ones(3) * origin

    if boxsize is not None:
      self.need_cut = True
    else:
      self.need_cut = False

    if boxsize is None and 'boxsize' in self.C:
      boxsize = numpy.ones(3) * self.C['boxsize']

    if boxsize is not None:
      self.boxsize[...] = numpy.ones(3) * boxsize
    else:
      self.boxsize = numpy.ones(3)
    self.periodic = periodic
    self.cosmology = Cosmology(M=self.C['OmegaM'], L=self.C['OmegaL'], h=self.C['h'])
    self.units = self.cosmology.units
    self.invalidate()

    self.schema('gas', 0, ['sml', 'mass'])
    self.schema('bh', 5, ['bhmass', 'bhmdot'])
    self.schema('halo', 1, ['mass'])
    self.schema('star', 4, ['mass', 'sft'])
 
  def read(self, ftypes, fids=None, numthreads=None):
    if self.need_cut:
      cut = Cut(origin=self.origin, size=self.boxsize)
    else:
      cut = None
    if fids is not None:
      snapnames = [self.snapname % i for i in fids]
    elif '%d' in self.snapname:
      snapnames = [self.snapname % i for i in range(self.C['Nfiles'])]
    else:
      snapnames = [self.snapname]
    snapshots = [Snapshot(snapname, self.format) for snapname in snapnames]

    for ftype in _ensurelist(ftypes):
      self.F[ftype].take_snapshots(snapshots, ptype=self.P[ftype], nthreads=numthreads, cut=cut)
      self.T[ftype] = self.F[ftype].zorder(ztree=True, thresh=32)
    self.invalidate()

  def frame(self, off=None, bgcolor=None, ax=None):
    from matplotlib.ticker import Formatter
    class MyFormatter(Formatter):
      def __init__(self, size, units, unit=None):
        if unit is None:
          MPC_h = size / units.MPC_h
          KPC_h = size / units.KPC_h
          if numpy.abs(MPC_h) > 1:
            self.unit = 'MPC/h'
            self.scale = 1 / units.MPC_h
          else:
            self.unit = 'KPC/h'
            self.scale = 1 / units.KPC_h
      def __call__(self, data, pos=None):
        value = data
        if pos == 'label':
          return '%.10g %s' % (value * self.scale, self.unit)
        else:
          return '%.10g' % (value * self.scale)

    l,r,b,t = self.extent
    if off is None:
      off = not ax.axison
    formatter = MyFormatter(self.size[0], self.units)
    if not off:
      if bgcolor is not None:
        ax.set_axis_bgcolor(bgcolor)
      ax.xaxis.set_major_formatter(formatter)
      ax.yaxis.set_major_formatter(formatter)
      ax.set_xticks(numpy.linspace(l, r, 5, endpoint=True))
      ax.set_yticks(numpy.linspace(b, t, 5, endpoint=True))
    else :
      if bgcolor is not None:
        ax.figure.set_facecolor(color)
      if not hasattr('scale', ax):
        ax.scale = self.drawscale(ax)
    ax.axison = not off
    xoffset = formatter(self.center[0], 'label')
    yoffset = formatter(self.center[1], 'label')
    zoffset = formatter(self.center[2], 'label')
    if not hasattr(ax, 'offsetlabel'):
      ax.offsetlabel = ax.text(0.1, 0.1, '', transform=ax.transAxes)
    ax.offsetlabel.set_text('%s %s %s' % (xoffset, yoffset, zoffset))
    ax.set_xlim(l, r)
    ax.set_ylim(b, t)
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
    return b
  def makeT(self, ftype, Xh = 0.76):
    """T will be in Kelvin"""
    gas =self.F[ftype]
    C = gas.cosmology
    gas['T'] = numpy.zeros(dtype='f4', shape=gas.numpoints)
    C.ie2T(ie = gas['ie'], ye = gas['ye'], Xh = Xh, out = gas['T'])
    gas['T'] *= C.units.TEMPERATURE

context = GaplotContext()
from gaepsi.tools import bindmethods as _bindmethods

def _before(args, kwargs):
  if 'ax' in kwargs:
    if kwargs['ax'] is None:
      kwargs['ax'] = pyplot.gca()

def _after(args, kwargs):
  if 'ax' in kwargs and pyplot.isinteractive():
    pyplot.show()

_bindmethods(context, locals(), _before, _after)

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


