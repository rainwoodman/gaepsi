import numpy
from numpy import sin, cos
from numpy import newaxis
from cosmology import Cosmology
from cosmology import WMAP7
from tools import sharedmem
from warnings import warn

def is_string_like(v):
  try: v + ''
  except: return False
  return True
def is_scalar_like(v):
  if numpy.isscalar(v): return True
  if isinstance(v, numpy.ndarray):
    if v.ndim == 0: return True
  return False

class Cut:
  def __init__(self, origin=None, center=None, size=None):
    if size is not None:
      self.size = numpy.ones(3) * size
      if origin is not None:
        self.center = self.size * 0.5 + origin
      elif center is not None:
        self.center = numpy.ones(3) * center
      else:
        self.center = self.size * 0.5
    else:
      self.center = None
      self.size = None

  @property
  def empty(self):
    return self.center is None

  @property
  def origin(self):
    return self.center - self.size * 0.5

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
    return numpy.asarray([self.center[axis] - self.size[axis] * 0.5, 
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
    mask = numpy.ones(dtype='?', shape = locations.shape[0])
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

  def __init__(self, components=None, numpoints = 0, dtype='f4'):
    """components is a dictionary of {component=>dtype}"""
    self.dict = {}
    self.numpoints = numpoints
    self['locations'] = numpy.zeros(shape = numpoints, dtype = (dtype, 3))
    if components is not None:
      for comp in components:
        self.dict[comp] = numpy.zeros(shape = numpoints, dtype = components[comp])

  def __len__(self):
    return self.numpoints

  @property
  def a(self):
    return 1. / (1. + self.redshift)
  @a.setter
  def a(self, value):
    self.redshift = 1. / a - 1.

  def init_from_snapshot(self, snapshot):
    if not 'OmegaM' in snapshot.C or not 'OmegaL' in snapshot.C or not 'h' in snapshot.C:
      warn("OmegaM, OmegaL, h not supported in snapshot, a default cosmology is used")
      self.comsology = WMAP7
    else:
      self.cosmology = Cosmology(K=0, M=snapshot.C['OmegaM'], L=snapshot.C['OmegaL'], h=snapshot.C['h'])

    if not 'redshift' in snapshot.C:
      warn('redshift not supported in snapshot, assuming redshift=0 (proper)')
      self.redshift = 0
    else:
      self.redshift = snapshot.C['redshift']

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
    starts = numpy.zeros(dtype = 'u8', shape = Nfile)
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
        snapshot[ptype, block] = numpy.array(self[comp][starts[i]:starts[i]+snapshot.C['N'][ptype]], dtype=dtype.base, copy=False)

        if save_and_clear:
          snapshot.save([block], ptype=ptype)
          snapshot.clear([block], ptype=ptype)
      #skip if the reader doesn't save the block

    with sharedmem.Pool(use_threads=True, np=nthreads) as pool:
      pool.map(work, list(range(Nfile)))

    if skipped_comps:
      print 'warning: blocks not supported in snapshot', skipped_comps

  def take_snapshots(self, snapshots, ptype, cut=None, nthreads=None):
    """ ptype can be a list of ptypes, in which case all particles of the types are loaded into the field """
    self.init_from_snapshot(snapshots[0])
    if numpy.isscalar(ptype):
       ptypes = [ptype]
    else:
       ptypes = ptype

    ptype = None

    self.numpoints = 0

      
    lengths = numpy.zeros(dtype='u8', shape=(len(snapshots), len(ptypes)))
    starts  = lengths.copy()

    with sharedmem.Pool(use_threads=True, np=nthreads) as pool:
      def work(i, snapshot):
        for j, ptype in enumerate(ptypes):
          mask = None
          if (ptype, 'pos') in snapshot:
            pos = snapshot[ptype, 'pos']
            if snapshot.C['N'][ptype] != 0 and cut is not None:
              mask = cut.select(pos)
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
      self.dict[comp] = numpy.zeros(shape = shape,
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
            if snapshot.C['N'][ptype] != 0 and cut is not None:
              mask = cut.select(pos)
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
      if index == 'x':
        return self.dict['locations'][:, 0]
      elif index == 'y':
        return self.dict['locations'][:, 1]
      elif index == 'z':
        return self.dict['locations'][:, 2]
      return self.dict[index]
    elif isinstance(index, slice):
      subfield = Field()
      start, stop, step = index.indices(self.numpoints)
      subfield.numpoints = (stop + step - 1 - start) / step
      for comp in self.names:
        subfield[comp] = self[comp][index]
      return subfield
    elif isinstance(index, numpy.ndarray) \
       and index.dtype == numpy.dtype('?'):
        subfield = Field()
        subfield.numpoints = index.sum()
        for comp in self.names:
          subfield[comp] = self[comp][index]
        return subfield
    elif not numpy.isscalar(index):
      subfield = Field()
      subfield.numpoints = len(index)
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
        value = numpy.repeat(value, self.numpoints)
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
    return 'Field(numpoints=%d, components=%s)' % (self.numpoints, 
            repr(d))

  def describe(self, index):
    if self.numpoints > 0:
      v = self[index]
      return dict(min=v.min(axis=0), max=v.max(axis=0))
    else:
      return dict(min=None, max=None)

  def dist(self, origin):
    d2 = ((self['locations'] - origin) ** 2).sum(axis=1)
    return d2 ** 0.5

  def smooth(self, weight='mass', NGB=0):
    """ smooth a field. when NGB<=0, a quick method is used to give
        a smoothing length estimated from the nearest tree node size; 
        the weight is not used.
        otherwise the sph kernel of nearest NGB particles is used to find
        a mass conserving smoothing length, the weight is used as the mass.
    """
    # important to first zorder the tree because it reorders the components.
    print self['locations'].shape
    tree = self.zorder(ztree=True)
    if weight is not None:
      weight = self[weight]
    else:
      # an 0d array is not chunked by the pool. 
      weight = numpy.asarray(1.0, dtype='f4')

    points = self['locations']
    try:
      sml = self['sml']
    except KeyError:
      self['sml'] = numpy.zeros(self.numpoints, 'f4')
      sml = self['sml']

    from cython._field import solve_sml
    
    def work(points, w, out): 
      solve_sml(points, w, self['locations'], numpy.atleast_1d(weight), out, tree, NGB)
    with sharedmem.Pool(use_threads=True) as pool:
      pool.starmap(work, pool.zipsplit((points, weight, sml), nchunks=1024))

  def rotate(self, angle, axis, origin):
    """angle is in degrees"""
    angle *= (3.14159/180)
    if axis == 2 or axis == 'z':
      M = numpy.matrix([[ cos(angle), -sin(angle), 0],
                  [ sin(angle), cos(angle), 0],
                  [ 0         ,          0, 1]], dtype='f4')
    if axis == 1 or axis == 'y':
      M = numpy.matrix([[ cos(angle), 0, -sin(angle)],
                  [ 0         ,          1, 0],
                  [ sin(angle), 0, cos(angle)]], dtype='f4')
    if axis == 0 or axis == 'x':
      M = numpy.matrix([[ 1, 0         ,          0],
                  [ 0, cos(angle), -sin(angle)],
                  [ 0, sin(angle), cos(angle)]], dtype='f4')

    self['locations'] -= origin
    self['locations'] = numpy.inner(self['locations'], M)
    self['locations'] += origin
    for comp in self.names:
      if comp != 'locations':
        if len(self[comp].shape) > 1:
          self[comp] = numpy.inner(self[comp], M)

  def redshift_distort(self, dir, vel=None):
    """ perform redshift distortion along direction dir, needs 'vel' and 'pos'
        if vel is None, field['vel'] is converted to peculiar velocity via multiplying by sqrt(1 / ( 1 + redshift)). A constant H calculated from redshift is used. The position are still given in comoving distance units, NOT the velocity unit. 
    """
    a = self.a
    if vel is None: vel = sqrt(a) * self['vel']
    H = self.cosmology.H(a = a)
    v = numpy.inner(vel, dir) / H
    self['locations'] += dir[newaxis,:] * v[:, newaxis] / a

  def unfold(self, M, boxsize):
    """ unfold the field position by transformation M
        the field shall be periodic. M is an
        list of column integer vectors of the shearing
        vectors. abs(det(M)) = 1
        the field has to be in a cubic box located from (0,0,0)
    """
    from tools.remap import remap

    pos = self['locations']
    pos /= boxsize
    newpos,newboxsize = remap(M, pos)
    newpos *= boxsize
    self['locations'] = newpos
    return newboxsize * boxsize

  def zorder(self, sort=True, ztree=False, thresh=128):
    """ calculate zkey (morton key) and return it.
        if sort is true, the field is sorted by zkey
        if ztree is false, return zkey, digitize, where digitize is
        an object converting from zkey and coordinates.
        if ztree is true, the field is permuted by zkey ordering,
        and a ZTree is returned.
        digitize object is used to convert pos to zkey and verse vica
        Notice that if sort or ztree is true, all previous reference
        to the field's components are invalid.
    """
    from cython import ztree as zt
    from cython import zorder as zo

    digitize = zo.Digitize.adapt(self['locations'])
    zkey = numpy.empty(self.numpoints, dtype=zo.zorder_dtype)
    with sharedmem.Pool(use_threads=True) as pool:
      def work(zkey, locations):
        digitize(locations, out=zkey)
      pool.starmap(work, pool.zipsplit((zkey, self['locations'])))

    if sort or ztree:
      # use sharemem.argsort, because it is faster
      arg = sharedmem.argsort(zkey)
      for comp in self.dict:
        # use sharemem.take, because it is faster
        self.dict[comp] = sharedmem.take(self.dict[comp], arg, axis=0)

      zkey = zkey[arg]
    if ztree:
      return zt.Tree(zkey=zkey, digitize=digitize, thresh=thresh)
    return zkey, digitize

