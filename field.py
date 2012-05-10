from numpy import isscalar
from numpy import ones,zeros
from numpy import append
from numpy import asarray
from numpy import atleast_1d
from ccode import sml
from ccode import peanohilbert
from numpy import sin,cos, matrix
from numpy import inner
from numpy import newaxis
from numpy import ndarray

from cosmology import Cosmology
from tools import sharedmem

def is_string_like(v):
  try: v + ''
  except: return False
  return True
def is_scalar_like(v):
  if isscalar(v): return True
  if isinstance(v, ndarray):
    if v.ndim == 0: return True
  return False

class Cut:
  def __init__(self, xcut=None, ycut=None, zcut=None, center=None, size=None):
    if xcut is not None:
      self.center = zeros(3)
      self.size = zeros(3)
      self['x'] = xcut
      self['y'] = ycut
      self['z'] = zcut
      return

    if size is not None:
      if is_scalar_like(size): size = ones(3) * atleast_1d(size)
      if center is not None:
        self.center = asarray(center[0:3])
      else:
        self.center = asarray(size) * 0.5
      self.size = asarray(size[0:3])
      return
    # uninitialized cut
    self.center = None
    self.size = None

  @property
  def empty(self):
    return self.center is None

  @property
  def origin(self):
    return self.center - self.size * 0.5
  @origin.setter
  def origin(self, value):
    value = asarray(value)
    self.center[:] = value + self.size * 0.5

  def take(self, cut):
    if cut is not None:
      if cut.empty:
        self.center = None
        self.size = None
      else:
        self.size = cut.size.copy()
        self.center = cut.center.copy()

  def __repr__(self):
    if self.center is None:
      return 'Cut()'
    return 'Cut(center=%s, size=%s)' % (repr(self.center), repr(self.size))

  def __getitem__(self, axis):
    if axis == 'x': axis = 0
    if axis == 'y': axis = 1
    if axis == 'z': axis = 2
    return asarray([self.center[axis] - self.size[axis] * 0.5, 
                    self.center[axis] + self.size[axis] * 0.5])

  def __setitem__(self, axis, value):
    if axis == 'x': axis = 0
    if axis == 'y': axis = 1
    if axis == 'z': axis = 2
    value = tuple(value)
    if len(value) != 2:
      raise ValueError("accepts only (lower, upper)")
    self.center[axis] = (value[0] + value[1]) * 0.5
    self.size[axis] = (value[1] - value[0])

  def select(self, locations):
    """return a mask of the locations in the cut"""
    if self.empty:
      return None
    mask = ones(dtype='?', shape = locations.shape[0])
    for axis in range(3):
      mask[:] &= (locations[:, axis] >= self[axis][0])
      mask[:] &= (locations[:, axis] < self[axis][1])
    return mask

class Field(object):
  def __init__(self, components=None, numpoints = 0, cut=None):
    """components is a dictionary of {component=>dtype}"""
    self.dict = {}
    self.cut = Cut()
    self.cut.take(cut)
    self.numpoints = numpoints
    self['locations'] = zeros(shape = numpoints, dtype = ('f4', 3))
    self.redshift = 0
    self.boxsize = 0
    if components is not None:
      for comp in components:
        self.dict[comp] = zeros(shape = numpoints, dtype = components[comp])

  def __len__(self):
    return self.numpoints

  @property
  def a(self):
    return 1. / (1. + self.redshift)
  @a.setter
  def a(self, value):
    self.redshift = 1. / a - 1.

  def init_from_snapshot(self, snapshot, cut=None):
    self.boxsize = snapshot.C['boxsize']
    self.cut.take(cut)
    self.cosmology = Cosmology(K=0, M=snapshot.C['OmegaM'], L=snapshot.C['OmegaL'], h=snapshot.C['h'])
    self.redshift = snapshot.C['redshift']

  def comp_to_block(self, comp):
    if comp == 'locations': return 'pos'
    return comp

  def dump_snapshots(self, snapshots, ptype):
    Nfile = len(snapshots)
    starts = zeros(dtype = 'u8', shape = Nfile)
    for i in range(Nfile):
      snapshot = snapshots[i]
      starts[i] = self.numpoints * i / Nfile
      snapshot.C.N[ptype] = self.numpoints * (i + 1) / Nfile - self.numpoints * i / Nfile
      tmp = snapshot.C.Ntot
      tmp[ptype] = self.numpoints
      snapshot.C['Ntot'] = tmp
      snapshot.C['Nfiles'] = Nfile
      snapshot.C['OmegaM'] = self.cosmology.M
      snapshot.C['OmegaL'] = self.cosmology.L
      snapshot.C['h'] = self.cosmology.h
      snapshot.C['boxsize'] = self.boxsize
      snapshot.C['redshift'] = self.redshift
    skipped_comps = set([])

    for i in range(Nfile):
      snapshot = snapshots[i]
      for comp in self:
        block = self.comp_to_block(comp)
        try:
          dtype = snapshot.reader.hash[block]['dtype']
        except KeyError:
          skipped_comps.update(set([comp]))
          continue
        if dtype.base is not self[comp].dtype:
          snapshot.P[ptype][block] = self[comp][starts[i]:starts[i]+snapshot.C.N[ptype]].astype(dtype.base)
        else:
          snapshot.P[ptype][block] = self[comp][starts[i]:starts[i]+snapshot.C.N[ptype]]
      #skip if the reader doesn't save the block
    if skipped_comps:
      print 'warning: blocks not supported in snapshot', skipped_comps

  def take_snapshots(self, snapshots, ptype, nthreads=None):
    self.init_from_snapshot(snapshots[0])

    self.numpoints = 0

    lengths = zeros(dtype='u8', shape=len(snapshots))
    starts  = lengths.copy()

    with sharedmem.Pool(use_threads=True, np=nthreads) as pool:
      def work(i, snapshot):
        snapshot.load(ptype = ptype, blocknames = ['pos'])
        if snapshot.C.N[ptype] != 0:
          mask = self.cut.select(snapshot.P[ptype]['pos'])
          if mask is not None:
            lengths[i] = mask.sum()
          else:
            lengths[i] = snapshot.C.N[ptype]
      pool.starmap(work, list(enumerate(snapshots)))
       
    starts[1:] = lengths.cumsum()[:-1]

    self.numpoints = lengths.sum()

    blocklist = []

    def resize(comp):
      shape = list(self[comp].shape)
      shape[0] = self.numpoints
      self.dict[comp] = zeros(shape = shape,
         dtype = self.dict[comp].dtype)

    resize('locations')

    for comp in self:
      if comp == 'locations': continue # skip locations it is handled differnently
      block = self.comp_to_block(comp)
      if not block in snapshots[0].reader:
        print block, 'is not supported in snapshot'
      else:
        blocklist.append((comp, block))
        resize(comp)

    with sharedmem.Pool(use_threads=True, np=nthreads) as pool:
      def work(snapshot, start, length):
        if length == 0: return
        pos, = snapshot.load(blocknames=['pos'], ptype=ptype)
        mask = None
        if snapshot.C.N[ptype] != 0:
          mask = self.cut.select(snapshot.P[ptype]['pos'])
          if mask is None:
            self['locations'][start:start+length] = pos[:]
          else:
            self['locations'][start:start+length] = pos[mask]
        del pos
        snapshot.clear('pos')
  
        for comp, block in blocklist:
          data, = snapshot.load(ptype = ptype, blocknames = [block])
          if mask is None:
            self[comp][start:start+length] = data[:]
          else:
            self[comp][start:start+length] = data[mask]
          del data
          snapshot.clear(self.comp_to_block(comp))
        del mask
      pool.starmap(work, zip(snapshots, starts, lengths))

  def add_snapshot(self, snapshot, ptype):
    """ """

    if snapshot.C['N'][ptype] == 0: return 0

    for comp in self:
      snapshot.load(ptype = ptype, blocknames = [self.comp_to_block(comp)])

    mask = self.cut.select(snapshot.P[ptype]['pos'])
    if mask is not None:
      add_points = mask.sum()
      if add_points == 0: return 0

      for comp in self:
        self.dict[comp] = append(self[comp], 
            snapshot.P[ptype][self.comp_to_block(comp)][mask], 
            axis = 0)
    else:
      add_points = snapshot.C['N'][ptype]
      for comp in self:
        self.dict[comp] = append(self[comp], 
            snapshot.P[ptype][self.comp_to_block(comp)], 
            axis = 0)

    self.numpoints = self.numpoints + add_points

    for comp in self:
      snapshot.clear(self.comp_to_block(comp))
    
    return add_points

  def __iter__(self):
    return iter(self.dict)
  def __str__(self) :
    return str(self.dict)
  def __getitem__(self, index):
    if type(index) is str:
      return self.dict[index]
    else:
      from numpy import repeat, array
      return repeat(array([index]), self.numpoints)

  def __setitem__(self, index, value):
    if is_scalar_like(value):
      value = ones(self.numpoints) * atleast_1d(value)
    if value.shape[0] != self.numpoints:
      raise ValueError("num of points of value doesn't match, %d != %d(new)" %( value.shape[0], self.numpoints))
    self.dict[index] = value
  
  def __delitem__(self, index):
    del self.dict[index]

  def __contains__(self, index):
    return index in self.dict

  def __repr__(self):
    d = {}
    for key in self.dict:
      d[key] = self.dict[key].dtype
    return 'Field(numpoints=%d, components=%s, cut=%s>' % (self.numpoints, 
            repr(d), repr(self.cut))

  def describe(self, index):
    if self.numpoints > 0:
      v = self[index]
      return dict(min=v.min(axis=0), max=v.max(axis=0))
    else:
      return dict(min=None, max=None)

  def dist(self, origin):
    d2 = ((self['locations'] - origin) ** 2).sum(axis=1)
    return d2 ** 0.5

  def smooth(self, weight='mass', NGB=32, tol=1e-5):
#    self.peano_reorder()
#    self['sml'] = sml(locations = self['locations'], mass = self['mass'], N=NGB)
    if weight is not None:
      expect = self[weight].mean() * NGB
    else:
      expect = NGB
    tree = self.zorder(tree=True)
    points = self['locations']
    sml = empty(self.numpoints, dtype='f4')
    def work(points, out): 
      ngb = tree.query_neighbours(points, NGB)
      neipos = self['locations'][ngb, :]
      if weight is not None:
        neiwei = self[weight][ngb] 
      else:
        neiwei = 1
      neipos -= points[:, newaxis, :]
      neipos **= 2
      dist = neipos.sum(axis=-1) ** 0.5
      del neipos
      hmax = dist[:, -1].copy()
      hmin = dist[:, 1].copy()
      while True:
        hmax *= 2
        mmax = 4 * pi / 3 *k0(dist / hmax[:, newaxis]).sum(axis=-1)
        if (mmax > expt).all(): break
      while True:
        hmin *= 0.5
        mmin = 4 * pi / 3 *k0(dist / hmin[:, newaxis]).sum(axis=-1)
        if (mmin < expt).all(): break
      while True:
        h = (hmin + hmax) * 0.5
        m = 4 * pi / 3 *k0(dist / h[:, newaxis]).sum(axis=-1)
        mask = m > expt
        hmax[mask] = h[mask]
        hmin[~mask] = h[~mask]
        if (1 - hmin/hmax > tol).all(): break
    with sharedmem.Pool(use_threads=True) as pool:
      pool.map(work, pool.split((points, out), nchunk=1024))

  def rotate(self, angle, axis, origin):
    """angle is in degrees"""
    angle *= (3.14159/180)
    if axis == 2 or axis == 'z':
      M = matrix([[ cos(angle), -sin(angle), 0],
                  [ sin(angle), cos(angle), 0],
                  [ 0         ,          0, 1]])
    if axis == 1 or axis == 'y':
      M = matrix([[ cos(angle), 0, -sin(angle)],
                  [ 0         ,          1, 0],
                  [ sin(angle), 0, cos(angle)]])
    if axis == 0 or axis == 'x':
      M = matrix([[ 1, 0         ,          0],
                  [ 0, cos(angle), -sin(angle)],
                  [ 0, sin(angle), cos(angle)]])

    self['locations'] -= origin
    self['locations'] = inner(self['locations'], M)
    self['locations'] += origin
    for comp in self.dict.keys():
      if comp != 'locations':
        if len(self[comp].shape) > 1:
          self[comp] = inner(self[comp], M)

  def redshift_distort(self, dir, vel=None):
    """ perform redshift distortion along direction dir, needs 'vel' and 'pos'
        if vel is None, field['vel'] is converted to peculiar velocity via multiplying by sqrt(1 / ( 1 + redshift)). A constant H calculated from redshift is used. The position are still given in comoving distance units, NOT the velocity unit. 
    """
    a = self.a
    if vel is None: vel = sqrt(a) * self['vel']
    H = self.cosmology.H(a = a)
    v = inner(vel, dir) / H
    self['locations'] += dir[newaxis,:] * v[:, newaxis] / a

  def unfold(self, M):
    """ unfold the field position by transformation M
        the field shall be periodic. M is an
        list of column integer vectors of the shearing
        vectors. abs(det(M)) = 1
        the field has to be in a cubic box located from (0,0,0)
    """
    from tools.remap import remap
    boxsize = self.boxsize

    pos = self['locations']
    pos /= boxsize
    newpos,newboxsize = remap(M, pos)
    newpos *= boxsize
    self['locations'] = newpos
    self.boxsize = newboxsize * boxsize

  def zorder(self, sort=True, ztree=False):
    """ fill in the ZORDER key of the field to ['_ZORDER'],
        and return it. 
        if sort is tree, the field is sorted by zorder
        if ztree is false, return zorder, scale 
        if ztree is true, the field is permuted into the zorder,
        and a ZTree is returned.
    """
    from ccode import ztree
    x, y, z = (self['locations'][:, 0],
              self['locations'][:, 1],
              self['locations'][:, 2])

    scale = ztree.Scale(x, y, z, bits=21)
    self['_ZORDER'] = ztree.zorder(x, y, z, scale=scale)
    if sort or ztree:
      # use sharemem.argsort, because it is faster
      arg = sharedmem.argsort(self['_ZORDER'])
      for comp in self.dict:
        self.dict[comp] = self.dict[comp][arg]
    if ztree:
      return ztree.Tree(zorder=self['_ZORDER'], scale=scale, thresh=30)
    return self['_ZORDER'], scale

  def peano_reorder(self):
    xyz = zeros(shape = (self.numpoints, 3), dtype='i4')
    pos=self['locations'] 
    min = pos.min(axis=0)
    max = pos.max(axis=0)
    dinv = (1<<19) / (max - min)
    xyz[:,:] = (pos - min[newaxis, :]) * dinv[newaxis, :]
    print min, max, dinv
    key = peanohilbert(xyz[:,0], xyz[:, 1], xyz[:,2])
    arg = key.argsort()
    for comp in self.dict:
      self.dict[comp] = self.dict[comp][arg]
    #return arg, key, xyz
