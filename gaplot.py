import numpy 
from gaepsi.snapshot import Snapshot
from gaepsi.field import Field, Cut
from gaepsi.readers import Reader
from gaepsi.compiledbase.camera import Camera
from gaepsi.compiledbase import _fast
from gaepsi.tools.meshmap import Meshmap
from gaepsi.tools import nl_, n_, normalize
import sharedmem
from gaepsi.cosmology import Cosmology
import warnings
import matplotlib
from matplotlib import cm

DEG = numpy.pi / 180.

def _ensurelist(a):
  if numpy.isscalar(a):
    return (a,)
  return a
def _fr10(n):
  """ base 10 frexp """
  exp = numpy.floor(numpy.log10(n))
  return n * 10 ** -exp, exp

def _image(color, luminosity=None, cmap=None, composite=False):
    """converts (color, luminosity) to an rgba image,
       if composite is False, directly reduce luminosity
       on the RGBA channel by luminosity,
       if composite is True, reduce the alpha channel.
       color and luminosity needs to be normalized/clipped to (0, 1), 
       otherwise will give nonsense results.
    """
    if cmap is None:
      if luminosity is not None:
        cmap = cm.coolwarm
      else:
        cmap = cm.gist_heat
    image = cmap(color, bytes=True)
    if luminosity is not None:
      luminosity = luminosity.copy()
      if composite:
        numpy.multiply(luminosity, 255.999, image[..., 3], casting='unsafe')
      else:
        numpy.multiply(image[...,:3], luminosity[..., None], image[..., :3], casting='unsafe')
    return image

image = _image

def addspikes(image, x, y, s, color):
  """deprecated method to directly add spikes to a raw image. 
     use context.scatter(fancy=True) instead
  """
  from matplotlib.colors import colorConverter
  color = numpy.array(colorConverter.to_rgba(color))
  for X, Y, S in zip(x, y, s):
    def work(image, XY, S, axis):
      S = int(S)
      if S < 1: return
      XY = numpy.int32(XY)
      start = XY[axis] - S
      end = XY[axis] + S + 1

      fga = ((1.0 - numpy.abs(numpy.linspace(-1, 1, end-start, endpoint=True))) * color[3])
      #pre multiply
      fg = color[0:3][None, :] * fga[:, None]

      if start < 0: 
        fga = fga[-start:]
        fg = fg[-start:]
        start = 0
      if end >= image.shape[axis]:
        end = image.shape[axis] - 1
        fga = fga[:end - start]
        fg = fg[:end - start]
 
      if axis == 0:
        sl = image[start:end, XY[1], :]
      elif axis == 1:
        sl = image[XY[0], start:end, :]

      bg = sl / 256.

      bga = bg[..., 3]
      bg = bg[..., 0:3] * bga[..., None]
      c = bg * (1-fga[:, None]) + fg
      ca = (bga * (1-fga) + fga)
      sl[..., 0:3] = c * 256
      sl[..., 3] = ca * 256
    work(image, (X, Y), S, 0)
    work(image, (X, Y), S, 1)

class GaplotContext(object):
  def __init__(self, shape = (600,600), thresh=(32, 64)):
    """ thresh is the fineness of the tree """
    self.default_axes = None
    self._format = None
    self._thresh = thresh
    # fields
    self.F = {}
    # ptypes
    self.P = {}
    # Trees
    self.T = {}
    # VT cache
    self.VT = {}
    # empty C avoiding __getattr__ recusive before first time use is called.
    self.C = {}

    self.periodic = False
    self.origin = numpy.array([0, 0, 0.])
    self.boxsize = numpy.array([0, 0, 0.])

    self._shape = shape
    self.camera = Camera(width=self.shape[0], height=self.shape[1])
    self.view(center=[0, 0, 0], size=[1, 1, 1], up=[0, 1, 0], dir=[0, 0, -1], fade=False, method='ortho')
    from gaepsi.cosmology import default
    self.cosmology = default
    
    # used to save the last state of plotting routines
    self.last = {}

  def __getattr__(self, attr):
    
    if attr in self.C:
      return self.C[attr]
    else: raise AttributeError('attribute %s not found' % attr)

  @property
  def U(self):
    return self.cosmology.units

  @property
  def shape(self):
    return self._shape
    
  def reshape(self, newshape):
    self._shape = (newshape[0], newshape[1])
    self.camera.shape = self._shape

  def __getitem__(self, ftype):
    return self.F[ftype]

  def __setitem__(self, ftype, value):
    if not isinstance(value, Field):
      raise TypeError('need a Field')
    self.F[ftype] = value
    self._rebuildtree(ftype)

  def image(self, color, luminosity=None, cmap=None, composite=False):
    # a convenient wrapper
    return _image(color, luminosity=luminosity, cmap=cmap, composite=composite)

  def _rebuildtree(self, ftype, thresh=None):
    from gaepsi.cython import fillingcurve as fc
    if thresh is None: thresh = self._thresh
    if (self.boxsize[...] == 0.0).all():
      self.boxsize[...] = self.F[ftype]['locations'].max(axis=0)
      scale = fc.scale(origin=self.F[ftype]['locations'].min(axis=0), boxsize=self.F[ftype]['locations'].ptp(axis=0))
    else:
      scale = fc.scale(origin=self.origin, boxsize=self.boxsize)
    zkey, scale = self.F[ftype].zorder(scale)
    self.T[ftype] = self.F[ftype].ztree(zkey, scale, minthresh=min(thresh), maxthresh=max(thresh))
    self.T[ftype].optimize()
    if ftype in self.VT:
      del self.VT[ftype]

  def print_png(self, *args, **kwargs):
    """ save the default figure to a png file """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(self.default_axes.figure)
    canvas.print_png(*args, **kwargs)

  def figure(self, dpi=200., figsize=None, axes=[0, 0, 1, 1.]):
    """ setup a default figure and a default axes """
    from matplotlib.figure import Figure
    if figsize is None:
      width = self.shape[0] / dpi
      height = self.shape[1] / dpi
    else:
      width, height = figsize
    figure = Figure((width, height), dpi=dpi)
    ax = figure.add_axes(axes)
    self.default_axes = ax
    return figure, ax

  def invalidate(self, *ftypes):
    """ clear the VT cache of a ftype """
    for ftype in ftypes:
      for key in list(self.VT.keys()):
        if key[0] == ftype:
          del self.VT[key]
    
  @property
  def extent(self):
    return self.camera.extent

  @property
  def default_camera(self):
    """ use camera instead """
    warnings.warn('default_camera deprecated. use camera instead')
    return self.camera

  def view(self, center=None, size=None, up=None, dir=None, method='ortho', fade=None):
    """ 
       if center is a field type, 
         center will be calcualted from that field type
         if size is None, size will be derived, too

       in all othercases, anything non-none will be overwritten,

    """
    if isinstance(center, basestring):
      ftype = center
      if self.F[ftype].numpoints > 0:
        center = self.T[ftype][0].pos + 0.5 * self.T[ftype][0].size
      if size is None:
        size = self.T[ftype][0].size

    if center is not None:
      self.center = numpy.ones(3) * center
    if size is not None:
      self.size = numpy.ones(3) * size
    if up is not None:
      self.up = numpy.ones(3) * up
    if dir is not None:
      self.dir = numpy.ones(3) * dir
    if fade is not None:
      self.fade = fade

    self._mkcamera(method, self.fade)

  def _mkcamera(self, method, fade):
    """ make a camera, method can be ortho or persp
        also set the fade property of the camera,
        determining whether the distance is used
        to simulate the brightness reduction.
    """

    target = self.center
    distance = self.size[2]

    self.camera.lookat(target=target,
       pos=target - self.dir * distance, up=self.up)
    if method == 'ortho':
      self.camera.ortho(extent=(-self.size[0] * 0.5, self.size[0] * 0.5, -self.size[1] * 0.5, self.size[1] * 0.5), near=distance - 0.5 * self.size[2], far=distance + 0.5 *self.size[2])
    elif method == 'persp':
      fov = numpy.arctan2(self.size[1] * 0.5, distance) * 2
      aspect = self.size[0] / self.size[1]
      self.camera.persp(fov=fov, aspect=aspect, near=distance - 0.5 * self.size[2], far=distance + 0.5*self.size[2])
    self.camera.fade = fade

  def _mkcameras(self, camera=None):
    if not camera:
      camera = self.camera
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

  def paint(self, ftype, color, luminosity, sml=None, camera=None, kernel=None):
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
    CCD = numpy.zeros(self.shape, dtype=('f8',2))

    if sml is None:
      if kernel is None: kernel='cube'
      tree = self.T[ftype]
      if color is not None:
        tree[color], = self._getcomponent(ftype, color)
      if luminosity is not None:
        tree[luminosity], = self._getcomponent(ftype, luminosity)

      for cam in self._mkcameras(camera):
        mask = cam.prunetree(tree)
        pos = tree['pos'][mask]
        size = tree['size'][mask]
        pos += size * 0.5
        x,y,z = pos.T
        sml = size[:, 0] * (0.5 * numpy.cross(cam.dir, cam.up).dot([1, 1, 1.]))
        c, l = None, None
        if color is not None: c = tree[color][mask]
        if luminosity is not None: l = tree[luminosity][mask]
        with sharedmem.Pool(use_threads=True) as pool:
          def work(x,y,z,sml,color,luminosity):
            _CCD = cam.paint(x,y,z,sml,color,luminosity, kernel=kernel)
            with pool.lock:
              CCD[...] += _CCD
          pool.starmap(work, pool.zipsplit((x,y,z,sml,c,l)))
    else:
      if kernel is None: kernel='spline'
      locations, color, luminosity, sml = self._getcomponent(ftype,
        'locations', color, luminosity, sml)
      x, y, z = locations.T
      for cam in self._mkcameras(camera):
        with sharedmem.Pool(use_threads=True) as pool:
          def work(x,y,z,sml,color,luminosity):
            _CCD = cam.paint(x,y,z,sml,color,luminosity, kernel=kernel)
            with pool.lock:
              CCD[...] += _CCD
          pool.starmap(work, pool.zipsplit((x,y,z,sml,color,luminosity)))
    C, L = CCD[...,0], CCD[...,1]
    return C/L, L
    
  def _getcomponent(self, ftype, *components):
    """
      _getcomponent(Field(), 'locations') 
      _getcomponent('gas', 'locations')
      _getcomponent('gas', [1, 2, 3, 4, 5])
      _getcomponent(Field(), [1, 2, 3, 4, 5]) 
      _getcomponent(numpy.zeros((10, 3)), 'locations')
      _getcomponent(numpy.zeros((10, 3)), numpy.zeros(10))
    """
    def one(component):
      if isinstance(component, basestring):
        if isinstance(ftype, Field):
          field = ftype
          return field[component]
        elif isinstance(ftype, basestring):
          field = self.F[ftype]
          return field[component]
        else:
          return ftype
      else:
        return component
    return [one(a) for a in components]

  def select(self, ftype, sml=0, camera=None):
    locations, = self._getcomponent(ftype, 'locations')
    x, y, z = locations.T
    mask = numpy.zeros(len(x), dtype='?')
    for cam in self._mkcameras(camera):
      with sharedmem.Pool(use_threads=True) as pool:
        def work(mask, x, y, z, sml):
          mask[:] |= (cam.mask(x, y, z, sml) != 0)
        pool.starmap(work, pool.zipsplit((mask, x, y, z, sml)))
    return mask

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
    locations, luminosity = self._getcomponent(ftype, 'locations', luminosity)
    x, y, z = locations.T
    for cam in self._mkcameras(camera):
      uvt = cam.transform(x, y, z)
      x = (uvt[:, 0] + 1.0) * 0.5 * self.shape[0]
      y = (uvt[:, 1] + 1.0) * 0.5 * self.shape[1]
      if radius is not None:
        mask = x >= -radius
        mask&= y >= -radius
        mask&= x <= self.shape[0]+radius
        mask&= y <= self.shape[1]+radius
        mask&= uvt[:,2] > -1
        mask&= uvt[:,2] < 1
      else:
        mask = numpy.ones(x.shape, dtype='?')
      X += [x[mask]]
      Y += [y[mask]]
      if luminosity is not None:
        d = ((locations - cam.pos)**2).sum(axis=-1)
        D += [d[mask]]
        L += [luminosity[mask]]
      else:
        I += [mask.nonzero()[0]]
    X = numpy.concatenate(X)
    Y = numpy.concatenate(Y)
    if luminosity is not None:
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
    if ax is None: ax=self.default_axes
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

  def drawbox(self, center, size, color=[0, 1., 0], ax=None):
    if ax is None: ax=self.default_axes
    center = numpy.asarray(center)
    color = numpy.asarray(color, dtype='f8')
    bbox = (numpy.mgrid[0:2, 0:2, 0:2].reshape(3, -1).T - 0.5) * size \
          + center[None, :]

    X, Y, B = self.transform(bbox, numpy.ones(len(bbox)), radius=None)
    l, r, b, t =self.extent
    X = X / self.shape[0] * (r - l) + l
    Y = Y / self.shape[1] * (t - b) + b

    from matplotlib.collections import LineCollection
    pairs = ((0,1), (2,3), (6,7), (4,5),
         (1,5), (5,7), (7,3), (3,1),
         (0,4), (4,6), (6,2), (2,0))
    lines = [((X[a], Y[a]), (X[b], Y[b])) for a, b in pairs]
    colors = numpy.ones((len(lines), 4))
    colors[:, 0:3] *= color
    colors[:, 3] *= n_([B[a] + B[b] for a, b in pairs]) * 0.7 + 0.3
    ax.add_collection(LineCollection(lines, 
          linewidth=0.5, colors=colors, antialiased=1))

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
    if ax is None: ax=self.default_axes
    realkwargs = dict(origin='lower', extent=self.extent)
    if luminosity is not None:
       realkwargs['cmap'] = cm.coolwarm
    else:
       realkwargs['cmap'] = cm.gist_heat
    realkwargs.update(kwargs)
    self.last['cmap'] = realkwargs['cmap']
    cmap = realkwargs['cmap']
    if len(color.shape) < 3:
      if not isinstance(color, normalize):
        color = n_(color)
      if luminosity is not None and not isinstance(luminosity, normalize):
        luminosity = n_(luminosity)
      
      im = self.image(color=color, luminosity=luminosity, cmap=cmap)
    else:
      im = color
    print im[:, :, 3].min()
    ax.imshow(im.swapaxes(0,1), **realkwargs)

    self.last['color'] = color

  def scatter(self, x, y, s, ax=None, fancy=False, **kwargs):
    """ takes CCD coordinate x, y and plot them to a data coordinate axes,
        s is the radius in points(72 points = 1 inch). 
        When fancy is True, apply a radient filter so that the 
        edge is blent into the background; better with marker='o' or marker='+'. """
    x, y, s = numpy.asarray([x, y, s])
    if ax is None: ax=self.default_axes
    l, r, b, t =self.extent
    X = x / self.shape[0] * (r - l) + l
    Y = y / self.shape[1] * (t - b) + b
    if not fancy:
      return ax.scatter(X, Y, s*s, **kwargs)
    
    from matplotlib.markers import MarkerStyle
    from matplotlib.patches import PathPatch
    from matplotlib.transforms import Affine2D
    def filter(image, dpi):
      # this is problematic if the marker is clipped.
      if image.shape[0] <=1 and image.shape[1] <=1: return image
      xgrad = 1.0 \
         - numpy.fabs(numpy.linspace(0, 2, 
            image.shape[0], endpoint=True) - 1.0)
      ygrad = 1.0 \
         - numpy.fabs(numpy.linspace(0, 2, 
            image.shape[1], endpoint=True) - 1.0)
      image[..., 3] *= xgrad[:, None] ** 1.3
      image[..., 3] *= ygrad[None, :] ** 1.3
      return image, 0, 0

    marker = kwargs.pop('marker', None)
    verts = kwargs.pop('verts', None)
    transform = kwargs.pop('transform', ax.transData)
    # to be API compatible
    if marker is None and not (verts is None):
        marker = (verts, 0)
        verts = None

    marker_obj = MarkerStyle(marker)
    path = marker_obj.get_path()

    objs = []
    m = transform.get_affine().get_matrix()
    sx = m[0, 0]
    sy = m[1, 1]
    for x,y,r in zip(X, Y, s):
      patch_transform = Affine2D().scale(r / 72. * sx, r / 72. * sy).translate(x, y)
      obj = PathPatch(path.transformed(patch_transform), transform=transform, **kwargs)
      obj.set_agg_filter(filter)
      obj.rasterized = True
      objs += [obj]
      ax.add_patch(obj)
    return objs

  def schema(self, ftype, types, components):
    """ loc dtype is the base dtype of the locations."""
    reader = Reader(self._format)
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

  def use(self, snapname, format, periodic=False, origin=[0,0,0.], boxsize=None, mapfile=None):
    self.snapname = snapname
    self._format = format

    try:
      snapname = self.snapname % 0
    except TypeError:
      snapname = self.snapname

    snap = Snapshot(snapname, self._format)

    self.C = snap.C
    self.origin[...] = numpy.ones(3) * origin

    if boxsize is not None:
      self.need_cut = True
    else:
      self.need_cut = False

    if mapfile is not None:
      self.map = Meshmap(mapfile)
    else:
      self.map = None

    if boxsize is None and 'boxsize' in self.C:
      boxsize = numpy.ones(3) * self.C['boxsize']

    if boxsize is not None:
      self.boxsize[...] = boxsize
    else:
      self.boxsize[...] = 1.0 

    self.periodic = periodic
    self.cosmology = Cosmology.from_snapshot(snap)
    self.redshift = self.C['redshift']

    self.schema('gas', 0, ['sml', 'mass'])
    self.schema('bh', 5, ['bhmass', 'bhmdot', 'id'])
    self.schema('halo', 1, ['mass'])
    self.schema('star', 4, ['mass', 'sft'])
 
  def saveas(self, ftypes, snapshots, np=None):
    for ftype in _ensurelist(ftypes):
      self.F[ftype].dump_snapshots(snapshots, ptype=self.P[ftype], np=np, save_and_clear=True)

  def read(self, ftypes, fids=None, np=None):
    if self.need_cut:
      cut = Cut(origin=self.origin, size=self.boxsize)
      if fids is None and self.map is not None:
        fids = self.map.cut2fid(cut)
    else:
      cut = None

    if fids is not None:
      snapnames = [self.snapname % i for i in fids]
    elif '%d' in self.snapname:
      snapnames = [self.snapname % i for i in range(self.C['Nfiles'])]
    else:
      snapnames = [self.snapname]

    def getsnap(snapname):
      try:
        return Snapshot(snapname, self._format)
      except IOError as e:
        warnings.warn('file %s skipped for %s' %(snapname, str(e)))
      return None

    snapshots = filter(lambda x: x is not None, sharedmem.map(getsnap, snapnames))
    
    rt = []
    for ftype in _ensurelist(ftypes):
      self.F[ftype].take_snapshots(snapshots, ptype=self.P[ftype], np=np, cut=cut)
      self._rebuildtree(ftype)
      self.invalidate(ftype)
      rt += [self[ftype]]

    if numpy.isscalar(ftypes):
      return rt[0]
    else:
      return rt

  def frame(self, axis=None, bgcolor=None, scale=None, ax=None):
    """scale can be a dictionary or True.
       properties in scale:
          (scale=None, color=None, fontsize=None, loc=8, pad=0.1, borderpad=0.5, sep=5)
    """
    if ax is None: ax=self.default_axes
    from matplotlib.ticker import Formatter
    from matplotlib.text import Text
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
    formatter = MyFormatter(self.size[0], self.U)
    if bgcolor is not None:
      ax.set_axis_bgcolor(bgcolor)
      ax.figure.set_facecolor(bgcolor)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.set_xticks(numpy.linspace(l, r, 5, endpoint=True))
    ax.set_yticks(numpy.linspace(b, t, 5, endpoint=True))

    if hasattr(ax, 'scale') and ax.scale is not None:
      ax.scale.remove()
      ax.scale = None

    if scale != False:
      if isinstance(scale, dict):
        ax.scale = self._updatescale(ax=ax, **scale)
      else:
        ax.scale = self._updatescale(ax=ax)
      ax.add_artist(ax.scale)

    if axis:
      ax.axison = True
    elif axis == False:
      ax.axison = False

    ax.set_xlim(l, r)
    ax.set_ylim(b, t)

  def _updatescale(self, ax, scale=None, color=None, fontsize=None, loc=8, pad=0.1, borderpad=0.5, sep=5, comoving=True):
    from matplotlib.patches import Rectangle
    from matplotlib.text import Text
    from mpl_toolkits.axes_grid.anchored_artists import AnchoredSizeBar
    if scale is None:
      l = (self.extent[1] - self.extent[0]) * 0.2
    else:
      l = scale
      # always first put comoving distance to l
      if not comoving:
        l *= (1. + self.C['redshift'])

    if not comoving:
      # prefer integral distance numbers
      # for the type of distance of choice(comoving or proper)
      l /= (1. + self.C['redshift'])

    n, e = _fr10(l)
    l = numpy.floor(n) * 10 ** e

    if l > 500 :
      l/=1000.0
      l = int(l+0.5)
      text = r"%g Mpc/h" % l
      l *= 1000.0
    else:
      text = r"%g Kpc/h" %l
 
    if not comoving:
      # but the bar is always drawn in comoving
      l *= (1. + self.C['redshift'])

    ret = AnchoredSizeBar(ax.transData, l, text, loc=loc,
      pad=pad, borderpad=borderpad, sep=sep, frameon=False)

    if color is not None:
      for r in ret.size_bar.findobj(Rectangle):
        r.set_edgecolor(color)
      for t in ret.txt_label.findobj(Text):
        t.set_color(color)
    if fontsize is not None:
      for t in ret.txt_label.findobj(Text):
        t.set_fontsize(fontsize)

    return ret

  def makeP(self, ftype, Xh=0.76, halo=False):
    """return the hydrogen Pressure * volume """
    gas = self.F[ftype]
    gas['P'] = numpy.empty(dtype='f4', shape=gas.numpoints)
    self.cosmology.ie2P(ie=gas['ie'], ye=gas['ye'], mass=gas['mass'], abundance=1, Xh=Xh, out=gas['P'])

  def makeT(self, ftype, Xh=0.76, halo=False):
    """T will be in Kelvin"""
    gas = self.F[ftype]
    if halo:
      gas['T'] = gas['vel'][:, 0] ** 2
      gas['T'] += gas['vel'][:, 1] ** 2
      gas['T'] += gas['vel'][:, 2] ** 2
      gas['T'] *= 0.5
      gas['T'] *= self.U.TEMPERATURE
    else:
      gas['T'] = numpy.empty(dtype='f4', shape=gas.numpoints)
      self.cosmology.ie2T(ie=gas['ie'], ye=gas['ye'], Xh=Xh, out=gas['T'])
      gas['T'] *= self.U.TEMPERATURE
    


context = GaplotContext()
from gaepsi.tools import bindmethods as _bindmethods

def _before(self, args, kwargs):
    import matplotlib.pyplot as pyplot
    if 'ax' in kwargs and self.default_axes is None:
      if kwargs['ax'] is None:
        kwargs['ax'] = pyplot.gca()

def _after(self, args, kwargs):
    import matplotlib.pyplot as pyplot
    if 'ax' in kwargs and self.default_axes is None and pyplot.isinteractive():
      pyplot.show()

_bindmethods(context, locals(), _before, _after, excludes=('default_camera',))

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


