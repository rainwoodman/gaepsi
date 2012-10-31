import warnings
import numpy 
import sharedmem

from gaepsi.snapshot import Snapshot
from gaepsi.field import Field
from gaepsi.readers import Reader
from gaepsi.meshindex import MeshIndex
from gaepsi.cosmology import Cosmology

from gaepsi.compiledbase import fillingcurve

def _ensurelist(a):
  if numpy.isscalar(a):
    return (a,)
  return a

class Store(object):
  def __init__(self, thresh=(32, 64)):
    self._format = None
    self._thresh = thresh
    # fields
    self.F = {}
    # ptypes
    self.P = {}
    # Trees
    self.T = {}

    # empty C avoiding __getattr__ recusive before first time use is called.
    self.C = {}

    self.periodic = False
    self.origin = numpy.array([0, 0, 0.])
    self.boxsize = numpy.array([0, 0, 0.])

    from gaepsi.cosmology import default
    self.cosmology = default

  def __getattr__(self, attr):
    
    if attr in self.C:
      return self.C[attr]
    else: raise AttributeError('attribute %s not found' % attr)

  @property
  def U(self):
    return self.cosmology.units

  def __contains__(self, ftype):
    return ftype in self.F

  def __getitem__(self, ftype):
    return self.F[ftype]

  def __setitem__(self, ftype, value):
    if not isinstance(value, Field):
      raise TypeError('need a Field')
    self.F[ftype] = value
    self._rebuildtree(ftype)

  @property 
  def periodic(self):
    return self._periodic

  @periodic.setter
  def periodic(self, value):
    if value:
      self._periodic = True
    else:
      self._periodic = False

  def _rebuildtree(self, ftype, thresh=None):
    if thresh is None: thresh = self._thresh
    if (self.boxsize[...] == 0.0).all():
      self.boxsize[...] = self.F[ftype]['locations'].max(axis=0)
      scale = fillingcurve.scale(origin=self.F[ftype]['locations'].min(axis=0), boxsize=self.F[ftype]['locations'].ptp(axis=0))
    else:
      scale = fillingcurve.scale(origin=self.origin, boxsize=self.boxsize)
    zkey, scale = self.F[ftype].zorder(scale)
    self.T[ftype] = self.F[ftype].ztree(zkey, scale, minthresh=min(thresh), maxthresh=max(thresh))
    # optimize is useless I believe
    # self.T[ftype].optimize()

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

  def use(self, snapname, format, periodic=False, origin=[0,0,0.], boxsize=None, mapfile=None, **kwargs):
    """ kwargs will override values saved in the snapshot file,
        only particles within origin and boxsize are loaded.
    """
    self.snapname = snapname
    self._format = format

    try:
      snapname = self.snapname % 0
    except TypeError:
      snapname = self.snapname

    snap = Snapshot(snapname, self._format)

    self.C = snap.C
    for item in kwargs:
      self.C[item] = kwargs[item]

    self.origin[...] = numpy.ones(3) * origin

    if boxsize is not None:
      self.need_cut = True
    else:
      self.need_cut = False

    if mapfile is not None:
      self.map = MeshIndex.fromfile(mapfile)
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
    try:
      self.redshift = self.C['redshift']
    except:
      self.redshift = 0.0

    self.schema('gas', 0, ['sml', 'mass'])
    self.schema('bh', 5, ['bhmass', 'bhmdot', 'id'])
    self.schema('halo', 1, ['mass'])
    self.schema('star', 4, ['mass', 'sft'])
 
  def saveas(self, ftypes, snapshots, np=None):
    for ftype in _ensurelist(ftypes):
      self.F[ftype].dump_snapshots(snapshots, ptype=self.P[ftype][0], np=np, save_and_clear=True, cosmology=self.cosmology)

  def read(self, ftypes, fids=None, np=None):
    if self.need_cut:
      if fids is None and self.map is not None:
        fids = self.map.cut(self.origin, self.boxsize)
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

    with sharedmem.Pool(use_threads=True) as pool:
      snapshots = filter(lambda x: x is not None, pool.map(getsnap, snapnames))
    
    rt = []
    for ftype in _ensurelist(ftypes):
      self.F[ftype].take_snapshots(snapshots, ptype=self.P[ftype], np=np, cut=cut)
      self._rebuildtree(ftype)
      rt += [self[ftype]]

    if numpy.isscalar(ftypes):
      return rt[0]
    else:
      return rt

    self.T[ftype] = self.F[ftype].ztree(zkey, scale, minthresh=min(thresh), maxthresh=max(thresh))
    # i suspect this optimize has been deprecated.
#    self.T[ftype].optimize()

  def unfold(self, M, ftype=None, center=None):
    """ unfold the field position by transformation M
        the field shall be periodic. M is an
        list of column integer vectors of the shearing
        vectors. abs(det(M)) = 1
        the field has to be in a cubic box located from (0,0,0)
    """
    assert self.periodic
    from gaepsi.compiledbase.geometry import Rotation, Cubenoid
    if center is None:
      cub = Cubenoid(M, self.origin, self.boxsize, center=0, neworigin=0)
    else:
      cub = Cubenoid(M, self.origin, self.boxsize, center=center)

    self.boxsize[...] = cub.newboxsize
    self.origin[...] = cub.neworigin
    self.periodic = False

    if ftype is None: ftypes = self.F.keys()
    else: ftypes = _ensurelist(ftype)
    for ftype in ftypes:
      locations, = self._getcomponent(ftype, 'locations')
      x, y, z = locations.T
      with sharedmem.Pool(use_threads=True) as pool:
        def work(x, y, z):
          rt = cub.apply(x, y, z)
          return (rt < 0).sum()
        badness = pool.starmap(work, pool.zipsplit((x, y, z))).sum()
      print badness

    for ftype in ftypes:
      self._rebuildtree(ftype)

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

