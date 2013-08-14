import warnings
import numpy 
import sharedmem
import matplotlib
from matplotlib import cm
from matplotlib.backends.backend_agg import FigureCanvasAgg

from gaepsi.tools import nl_, n_, normalize
from gaepsi.store import *
from gaepsi.tools import loadconfig
from gaepsi.compiledbase.camera import Camera
from gaepsi.compiledbase.ztree import TreeProperty

try:
  from gaepsi.tools.analyze import HaloCatalog
  from gaepsi.tools.analyze import BHDetail
  from gaepsi.tools.analyze import profile
except:
  pass

DEG = numpy.pi / 180.

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
     s is pixels
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

      #bga = bg[..., 3]
      bga = 1.0
      #bg = bg[..., 0:3] * bga[..., None]
      c = bg * (1-fga[:, None]) + fg
      ca = (bga * (1-fga) + fga)
      sl[..., 0:3] = c * 256
      #sl[..., 3] = ca * 256
    work(image, (X, Y), S, 0)
    work(image, (X, Y), S, 1)

class GaplotContext(Store):
  def __init__(self, snapname=None,
          format=None, periodic=None, origin=[0, 0, 0.], boxsize=None, 
          mapfile=None,
          shape = (600,600), thresh=(32, 64), 
          **kwargs):
    """ thresh is the fineness of the tree 
        if snapname is given, call use immediately
    """

    Store.__init__(self, snapname, format, periodic, origin, boxsize, mapfile,
            shape, thresh, **kwargs)
    self.default_axes = None

    self.camera = Camera(width=shape[0], height=shape[1])
    self.reshape(shape)
    self.view(center=[0, 0, 0], size=[1, 1, 1], up=[0, 1, 0], dir=[0, 0, -1], fade=False, method='ortho')
    
    # used to save the last state of plotting routines
    self.last = {}

  @property
  def shape(self):
    return self._shape
    
  @property
  def CCD(self):
    return self._CCD

  def attach(self, CCD):
    newshape = CCD.shape
    self._shape = (newshape[0], newshape[1])
    self.camera.shape = self._shape
    self._CCD = CCD

  def reshape(self, newshape):
    self._shape = (newshape[0], newshape[1])
    self.camera.shape = self._shape
    self._CCD = numpy.zeros(self.shape, dtype=('f4',2))

  def image(self, color, luminosity=None, cmap=None, composite=False):
    # a convenient wrapper
    return _image(color, luminosity=luminosity, cmap=cmap, composite=composite)

  def savefig(self, *args, **kwargs):
    canvas = FigureCanvasAgg(self.default_axes.figure)
    self.default_axes.figure.savefig(*args, **kwargs)

  def newaxes(self, figure, axes=[0, 0, 1, 1.]):
    """ setup a default axes """
    dpi = figure.dpi
    width = 1.0 * self.shape[0] / dpi
    height = 1.0 * self.shape[1] / dpi
    figure.set_figheight(height)
    figure.set_figwidth(width)
    ax = figure.add_axes(axes)
    self.default_axes = ax
    return ax

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

  def paint(self, ftype, color, luminosity, sml=None, camera=None, kernel=None,
          preserve=False, np=None):
    """ paint field to CCD, returns
        C, L where
          C is the color of the pixel
          L is the exposure of the pixel
        Notice that if color is None, 
          C will be undefined,
          L will still be the exposure.
        the return values can be normalized by
        nl_ or n_, then feed to imshow
        if preserve is True, do not clean the CCD and return None
    """
    CCD = self.CCD
    if not preserve: CCD[...] = 0

    if isinstance(ftype, Field):
      self['__temp__'] = ftype
      ftype = '__temp__'
      self.buildtree('__temp__')
    tree = self.T[ftype]
    if kernel is None: kernel='spline'
    locations, color, luminosity, sml = self._getcomponent(ftype,
      'locations', color, luminosity, sml)

    if sml is None:
      warnings.warn('sml is None and the on-the-fly calculation is buggy, will run field.smooth first, sml is overwritten')
      self[ftype].smooth(tree)
      sml = self[ftype]['sml']

    x, y, z = locations.T
    for cam in self._mkcameras(camera):
      with sharedmem.TPool(np=self.np) as pool:
        cams = cam.divide(int(pool.np ** 0.5 * 2 + 1), int(pool.np ** 0.5 * 2 + 1))
        def work(cam, offx, offy):
          smallCCD = numpy.zeros(cam.shape, dtype=('f8', 2))
          cam.paint(x,y,z,sml,color,luminosity, kernel=kernel, out=smallCCD, tree=self.T[ftype])
          # no race condition here. cameras are independent.
          CCD[offx:offx + cam.shape[0], offy:offy+cam.shape[1], :] += smallCCD
        pool.starmap(work, cams.reshape(-1, 3))

    if not preserve:
        C, L = CCD[...,0], CCD[...,1]
        C = C / L
        return C, L
    else:
        return None
    
  def paint2(self, ftype, color, luminosity, camera=None, kernel=None, dtype='f8'):
    """ paint field to CCD, (this paints the tree nodes)
         returns
        C, L where
          C is the color of the pixel
          L is the exposure of the pixel
        Notice that if color is None, 
          C will be undefined,
          L will still be the exposure.
        the return values can be normalized by
        nl_ or n_, then feed to imshow
    """
    raise "Fix this."
    CCD = numpy.zeros(self.shape, dtype=(dtype,2))

    tree = self.T[ftype]
    if kernel is None: kernel='spline'
    color, luminosity= self._getcomponent(ftype,
      color, luminosity)

    if color is None:
      color = 1.0
    if luminosity is None:
      luminosity = 1.0

    colorp = TreeProperty(tree, color)
    luminosityp = TreeProperty(tree, luminosity)

    for cam in self._mkcameras(camera):
      with sharedmem.TPool(np=self.np) as pool:
        cams = cam.divide(int(pool.np ** 0.5 * 2), int(pool.np ** 0.5 * 2))
        def work(cam, offx, offy):
          mask = cam.prunetree(tree)
          x, y, z = tree['pos'][mask].T
          sml = tree['size'][mask][:, 0].copy() * 2
          luminosity = luminosityp[mask]
          color = colorp[mask]
          smallCCD = numpy.zeros(cam.shape, dtype=(dtype, 2))
          cam.paint(x,y,z,sml,color,luminosity, kernel=kernel, out=smallCCD)
          CCD[offx:offx + cam.shape[0], offy:offy+cam.shape[1], :] += smallCCD
        pool.starmap(work, cams.reshape(-1, 3))

    C, L = CCD[...,0], CCD[...,1]
    C[...] /= L
    return C, L

  def select(self, ftype, sml=0, camera=None):
    """ return a mask whether particles are in the camera """
    locations, = self._getcomponent(ftype, 'locations')
    x, y, z = locations.T
    mask = numpy.zeros(len(x), dtype='?')
    for cam in self._mkcameras(camera):
      with sharedmem.TPool(np=self.np) as pool:
        chunksize = 1024 * 1024
        def work(i):
          sl = slice(i, i + chunksize)
          mask[sl] |= (cam.mask(x[sl], y[sl], z[sl], sml[sl]) != 0)
        pool.map(work, range(0, len(mask), chunksize))
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
    for cam in self._mkcameras(camera):
      x, y, z = locations.T
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
        
        Notice that the convention of axes is different from
        pyplot.imshow. Here the first dimension is the horizontal
        and the second dimension is the vertical.
        Notice that the origin is also at the lower(thus a traditional
        x, y plot), rather than upper as in pyplot.imshow

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
    ax.imshow(im.swapaxes(0,1), **realkwargs)

    self.last['color'] = color

  def scatter(self, x, y, s, ax=None, fancy=False, **kwargs):
    """ takes CCD coordinate x, y and plot them to a data coordinate axes,
        s is the radius in data units. 
        When fancy is True, apply a radient filter so that the 
        edge is blent into the background; better with marker='o' or marker='+'. """
    x, y, s = numpy.asarray([x, y, s])
    if ax is None: ax=self.default_axes
    l, r, b, t =self.extent
    X = x / self.shape[0] * (r - l) + l
    Y = y / self.shape[1] * (t - b) + b
    
    from matplotlib.markers import MarkerStyle
    from matplotlib.patches import PathPatch
    from matplotlib.collections import PathCollection
    from matplotlib.transforms import Affine2D, IdentityTransform
    def filter(image, dpi):
      # this is problematic if the marker is clipped.
      if image.shape[0] <=1 and image.shape[1] <=1: return image
      xgrad = 1.0 \
         - numpy.fabs(numpy.linspace(0, 2, 
            image.shape[0], endpoint=True) - 1.0)
      ygrad = 1.0 \
         - numpy.fabs(numpy.linspace(0, 2, 
            image.shape[1], endpoint=True) - 1.0)
      image[..., 3] *= xgrad[:, None] ** 0.5
      image[..., 3] *= ygrad[None, :] ** 0.5
      return image, 0, 0

    marker = kwargs.pop('marker', None)
    verts = kwargs.pop('verts', None)
    # to be API compatible
    if marker is None and not (verts is None):
        marker = (verts, 0)
        verts = None

    objs = []
    color = kwargs.pop('color', None)
    edgecolor = kwargs.pop('edgecolor', None)
    linewidth = kwargs.pop('linewidth', kwargs.pop('lw', None))

    marker_obj = MarkerStyle(marker)
    if not marker_obj.is_filled():
        edgecolor = color

    for x,y,r in numpy.nditer([X, Y, s], flags=['zerosize_ok']):
      path = marker_obj.get_path().transformed(
         marker_obj.get_transform().scale(r).translate(x, y))
      obj = PathPatch(
                path,
                facecolor = color,
                edgecolor = edgecolor,
                linewidth = linewidth,
                transform = ax.transData,
              )
      obj.set_alpha(1.0)
      if fancy:
        obj.set_agg_filter(filter)
        obj.rasterized = True
      objs += [obj]
      ax.add_artist(obj)
    ax.autoscale_view()

    return objs

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
      try:
        ax.scale.remove()
      except:
        pass
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
    h = self.C['h']
    if scale is None:
      l = (self.extent[1] - self.extent[0]) * 0.2
    else:
      l = scale
      # always first put comoving distance to l
      if not comoving:
        l *= (1. + self.C['redshift'])
        l /= h

    if not comoving:
      # prefer integral distance numbers
      # for the type of distance of choice(comoving or proper)
      l /= (1. + self.C['redshift'])
      l *= h
      unit = ''
    else:
      unit = '/h'

    n, e = _fr10(l)
    l = numpy.floor(n) * 10 ** e

    if l > 500 :
      l/=1000.0
      l = int(l+0.5)
      text = r"%g Mpc%s" % (l, unit)
      l *= 1000.0
    else:
      text = r"%g Kpc%s" % (l, unit)
 
    if not comoving:
      # but the bar is always drawn in comoving
      l *= (1. + self.C['redshift'])
      l /= h

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

context = GaplotContext()
from gaepsi.tools import bindmethods as _bindmethods

def _before(self, args, kwargs):
    import matplotlib.pyplot as pyplot
    if 'figurefunc' in kwargs and pyplot.isinteractive():
      kwargs['figurefunc'] = pyplot.figure 
    if 'ax' in kwargs and self.default_axes is None:
      if kwargs['ax'] is None:
        kwargs['ax'] = pyplot.gca()

def _after(self, args, kwargs):
    import matplotlib.pyplot as pyplot
    if 'ax' in kwargs and pyplot.isinteractive():
      pyplot.draw()
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


