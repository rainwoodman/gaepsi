from numpy import isscalar
from numpy import ones,zeros, empty
from numpy import append
from numpy import asarray
from numpy import atleast_1d
from numpy import sin,cos, matrix
from numpy import inner
from numpy import newaxis
from numpy import ndarray

from cosmology import Cosmology
from tools import sharedmem
from ccode import k0
from warnings import warn

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
  @staticmethod
  def from_recarray(recarray):
    field = Field(numpoints = len(recarray))
    for name in recarray.dtype.fields:
      field[name] = recarray[name]
    return field

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
    if not 'boxsize' in snapshot.C:
      warn("boxsize not supported in snapshot")
    else:
      self.boxsize = snapshot.C['boxsize']
    if not 'OmegaM' in snapshot.C or not 'OmegaL' in snapshot.C or not 'h' in snapshot.C:
      warn("OmegaM, OmegaL, h not supported in snapshot")
    else:
      self.cosmology = Cosmology(K=0, M=snapshot.C['OmegaM'], L=snapshot.C['OmegaL'], h=snapshot.C['h'])
    if not 'redshift' in snapshot.C:
      warn('redshift not supported in snapshot')
    else:
      self.redshift = snapshot.C['redshift']
    self.cut.take(cut)

  def comp_to_block(self, comp):
    if comp == 'locations': return 'pos'
    return comp

  def dump_snapshots(self, snapshots, ptype, save_and_clear=False, nthreads=None):
    """ dump field into snapshots.
        if save_and_clear is True, immediately save the file and clear the snapshot object,
        using less memory.
        otherwise, leave the data in memory in snapshot object.
    """
    Nfile = len(snapshots)
    starts = zeros(dtype = 'u8', shape = Nfile)
    for i in range(Nfile):
      snapshot = snapshots[i]
      starts[i] = self.numpoints * i / Nfile
      snapshot.C['N'][ptype] = self.numpoints * (i + 1) / Nfile - self.numpoints * i / Nfile
      tmp = snapshot.C['Ntot']
      tmp[ptype] = self.numpoints
      snapshot.C['Ntot'] = tmp
      snapshot.C['Nfiles'] = Nfile
      snapshot.C['OmegaM'] = self.cosmology.M
      snapshot.C['OmegaL'] = self.cosmology.L
      snapshot.C['h'] = self.cosmology.h
      snapshot.C['boxsize'] = self.boxsize
      snapshot.C['redshift'] = self.redshift
    skipped_comps = set([])

    def work(i):
      snapshot = snapshots[i]
      if save_and_clear:
        snapshot.create_structure()

      for comp in self.names:
        block = self.comp_to_block(comp)
        try:
          dtype = snapshot.reader.hash[block]['dtype']
        except KeyError:
          skipped_comps.update(set([comp]))
          continue
        snapshot[ptype, block] = array(self[comp][starts[i]:starts[i]+snapshot.C['N'][ptype]], dtype=dtype.base, copy=False)

        if save_and_clear:
          snapshot.save([block], ptype=ptype)
          snapshot.clear([block], ptype=ptype)
      #skip if the reader doesn't save the block

    with sharedmem.Pool(use_threads=True, np=nthreads) as pool:
      pool.map(work, list(range(Nfile)))

    if skipped_comps:
      print 'warning: blocks not supported in snapshot', skipped_comps

  def take_snapshots(self, snapshots, ptype, nthreads=None):
    """ ptype can be a list of ptypes, in which case all particles of the types are loaded into the field """
    self.init_from_snapshot(snapshots[0])
    if isscalar(ptype):
       ptypes = [ptype]
    else:
       ptypes = ptype

    ptype = None

    self.numpoints = 0

      
    lengths = zeros(dtype='u8', shape=(len(snapshots), len(ptypes)))
    starts  = lengths.copy()

    with sharedmem.Pool(use_threads=True, np=nthreads) as pool:
      def work(i, snapshot):
        for j, ptype in enumerate(ptypes):
          mask = None
          if (ptype, 'pos') in snapshot:
            pos = snapshot[ptype, 'pos']
            if snapshot.C['N'][ptype] != 0:
              mask = self.cut.select(pos)
          if mask is not None:
            lengths[i, j] = mask.sum()
          else:
            lengths[i, j] = snapshot.C['N'][ptype]
      pool.starmap(work, list(enumerate(snapshots)))

    starts.flat[1:] = lengths.cumsum()[:-1]

    self.numpoints = lengths.sum()

    blocklist = []

    def resize(comp):
      shape = list(self[comp].shape)
      shape[0] = self.numpoints
      self.dict[comp] = zeros(shape = shape,
         dtype = self.dict[comp].dtype)

    if (ptypes[0], 'pos') in snapshots[0]:
      resize('locations')

    for comp in self.names:
      if comp == 'locations': continue # skip locations it is handled differnently
      block = self.comp_to_block(comp)

      blocklist.append((comp, block))
      resize(comp)

    #  if not (ptypes[0], block) in snapshots[0]:
    #    resize(comp)
    #    if block == 'mass':
    #      self[comp][:] = snapshots[0].header['mass'][ptype]
    #    else:
    #      print block, 'is not supported in snapshot'
    #  else:

    with sharedmem.Pool(use_threads=True, np=nthreads) as pool:
      def work(snapshot, start, length):
        for j, ptype in enumerate(ptypes):
          if length[j] == 0: continue
          mask = None
          if (ptype, 'pos') in snapshot:
            pos = snapshot[ptype, 'pos']
            if snapshot.C['N'][ptype] != 0:
              mask = self.cut.select(pos)
            if mask is None:
              self['locations'][start[j]:start[j]+length[j]] = pos[:]
            else:
              length0 = mask.sum()
              self['locations'][start[j]:start[j]+length[j]] = pos[mask]
            del pos
            del snapshot[ptype, 'pos']
  
          for comp, block in blocklist:
            if not (ptype, block) in snapshot:
              if block != 'mass':
                warn('ptype %d, %s not in snapshot file' % (ptype, block))
              else:
                self[comp][start[j]:start[j]+length[j]] = snapshot.header['mass'][ptype]
              continue
            data = snapshot[ptype, block]
            if mask is None:
              self[comp][start[j]:start[j]+length[j]] = data[:]
            else:
              self[comp][start[j]:start[j]+length[j]] = data[mask]
            del data
            del snapshot[ptype, block]
          del mask
      pool.starmap(work, zip(snapshots, starts, lengths))

  def __iter__(self):
    i = 0
    while True:
      (yield self[i])
      i = i + 1
      if i == self.numpoints: raise StopIteration
     
  def __str__(self) :
    return str(self.dict)

  @property
  def names(self):
    return self.dict.keys()

  def __getitem__(self, index):
    if isinstance(index, basestring):
      return self.dict[index]
    elif isinstance(index, slice):
      subfield = Field()
      start, stop, step = index.indices(self.numpoints)
      subfield.numpoints = (stop + step - 1 - start) / step
      for comp in self.names:
        subfield[comp] = self[comp][index]
      return subfield
    else:
      result = {}
      for comp in self.names:
        result[comp] = self[comp][index]
      return result
    
  def __setitem__(self, index, value):
    if isinstance(index, basestring):
      if is_scalar_like(value):
        value = ones(self.numpoints) * atleast_1d(value)
      if value.shape[0] != self.numpoints:
        raise ValueError("num of points of value doesn't match, %d != %d(new)" %( value.shape[0], self.numpoints))
      self.dict[index] = value

    elif isinstance(index, slice):
      raise IndexError("not supported setting a slice")
    else:
      raise IndexError("not supported setting a arbitrary index")
  
  def __delitem__(self, index):
    if isinstance(index, basestring):
      del self.dict[index]
    elif isinstance(index, slice):
      raise IndexError("not supported deleting a slice")
    else:
      raise IndexError("not supported deleting a arbitrary index")

  def __contains__(self, index):
    if isinstance(index, basestring):
      return index in self.dict
    else:
      return index >= 0 and index < self.numpoints

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

  def smooth(self, weight='mass', NGB=0, tol=1e-5):
    """ smooth a field. when NGB<=0, a quick method is used to give
        a smoothing length estimated from the nearest tree node size; 
        the weight is not used.
        otherwise the sph kernel of nearest NGB particles is used to find
        a mass conserving smoothing length, the weight is used as the mass.
    """
    # important to first zorder the tree because it reorders the components.
    tree = self.zorder(ztree=True)
    if weight is not None:
      weight = self[weight]
    else:
      # an 0d array is not chunked by the pool. 
      weight = asarray(1.0, dtype='f4')

    points = self['locations']
    sml = self['sml']
    from ccode._field import solve_sml
    
    def work(points, w, out): 
      solve_sml(points, w, self['locations'], atleast_1d(weight), out, tree, NGB)
    with sharedmem.Pool(use_threads=True) as pool:
      pool.starmap(work, zip(*pool.split((points, weight, sml), nchunks=1024)))

  def rotate(self, angle, axis, origin):
    """angle is in degrees"""
    angle *= (3.14159/180)
    if axis == 2 or axis == 'z':
      M = matrix([[ cos(angle), -sin(angle), 0],
                  [ sin(angle), cos(angle), 0],
                  [ 0         ,          0, 1]], dtype='f4')
    if axis == 1 or axis == 'y':
      M = matrix([[ cos(angle), 0, -sin(angle)],
                  [ 0         ,          1, 0],
                  [ sin(angle), 0, cos(angle)]], dtype='f4')
    if axis == 0 or axis == 'x':
      M = matrix([[ 1, 0         ,          0],
                  [ 0, cos(angle), -sin(angle)],
                  [ 0, sin(angle), cos(angle)]], dtype='f4')

    self['locations'] -= origin
    self['locations'] = inner(self['locations'], M)
    self['locations'] += origin
    for comp in self.names:
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

  def zorder(self, sort=True, ztree=False, thresh=128):
    """ fill in the ZORDER key and return it. 
        if sort is tree, the field is sorted by zorder
        if ztree is false, return zorder, scale 
        if ztree is true, the field is permuted into the zorder,
        and a ZTree is returned.
    """
    from ccode import ztree as zt
    x, y, z = (self['locations'][:, 0],
              self['locations'][:, 1],
              self['locations'][:, 2])

    scale = zt.Scale(x, y, z, bits=21)
    zorder = empty(self.numpoints, dtype='i8')
    with sharedmem.Pool(use_threads=True) as pool:
      def work(zorder, locations):
        x, y, z = locations[:, 0], locations[:, 1], locations[:, 2]
        zt.zorder(x, y, z, scale=scale, out=zorder)
      pool.starmap(work, zip(*pool.split((zorder, self['locations']))))

    if sort or ztree:
      # use sharemem.argsort, because it is faster
      arg = sharedmem.argsort(zorder)
      with sharedmem.Pool(use_threads=True) as pool:
        def work(key):
          self.dict[key] = self.dict[key].take(arg, axis=0)
        pool.map(work, list(self.dict.keys()))

      zorder = zorder[arg]
    if ztree:
      return zt.Tree(zorder=zorder, scale=scale, thresh=thresh)
    return zorder, scale

