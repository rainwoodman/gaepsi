import numpy
from numpy import sin, cos
from numpy import newaxis
import sharedmem
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

def filter(snap, ptype, origin, boxsize):
  if snap.C['N'][ptype] == 0: return None, 0
  if origin is None or boxsize is None:
    return None, snap.C['N'][ptype]
  tail = origin + boxsize
  pos = snap[ptype, 'pos']
  iter = numpy.nditer([pos[:, 0], pos[:, 1], pos[:, 2], None],
        op_dtypes=[None, None, None, '?'],
        op_flags=[['readonly']] * 3 + [['writeonly', 'allocate']],
        flags=['external_loop', 'buffered'])
  for x, y, z, m in iter:
    m[...] = \
      (x >= origin[0]) & (y >= origin[1]) & (z >= origin[2]) \
    & (x <= tail  [0]) & (y <= tail  [1]) & (z <= tail  [2])
    
  return iter.operands[3], iter.operands[3].sum()

def select(snap, ptype, block, mask):
  if mask is None:
    result = snap[ptype, block]
  else:
    result = snap[ptype, block][mask]
  del snap[ptype, block]
  return result

class Field(object):
  @staticmethod
  def from_recarray(recarray, locations='pos'):
    field = Field(numpoints=len(recarray), locations=locations)
    for name in recarray.dtype.fields:
      field[name] = recarray[name]
    return field

  def __init__(self, components=None, numpoints=0, locations='pos'):
    """components is a dictionary of {component=>dtype}"""
    self.dict = {}
    self.numpoints = int(numpoints)
    self.locations = locations

    if components is not None:
      for comp in components:
        self.dict[comp] = numpy.zeros(shape=numpoints, dtype=components[comp])

  def todict(self):
    d = {}
    for comp in self.names:
      d[comp] = self[comp]
    return d

  def __len__(self):
    return self.numpoints

  def dump_snapshots(self, snapshots, ptype, save_and_clear=False, C=None, np=None):
    """ dump field into snapshots.
        if save_and_clear is True, immediately save the file and clear the snapshot object,
        using less memory.
        otherwise, leave the data in memory in snapshot object. and only the header is written.
        C is the template used for the snapshot headers.
    """
    Nfile = len(snapshots)
    starts = numpy.zeros(dtype = 'u8', shape = Nfile)
    for i in range(Nfile):
      snapshot = snapshots[i]
      if C is not None:
        snapshot.C[...] = C
      starts[i] = self.numpoints * i / Nfile
      snapshot.C['N'][ptype] = self.numpoints * (i + 1) / Nfile - self.numpoints * i / Nfile
      tmp = snapshot.C['Ntot']
      tmp[ptype] = self.numpoints
      snapshot.C['Ntot'] = tmp
      snapshot.C['Nfiles'] = Nfile
    skipped_comps = set([])

    def work(i):
      snapshot = snapshots[i]
      if save_and_clear:
        snapshot.create_structure()

      for comp in self.names:
        try:
          dtype = snapshot.reader[comp].dtype
        except KeyError:
          skipped_comps.update(set([comp]))
          continue
        snapshot[ptype, comp] = numpy.array(self[comp][starts[i]:starts[i]+snapshot.C['N'][ptype]], dtype=dtype.base, copy=False)

        if save_and_clear:
          snapshot.save(comp, ptype)
          snapshot.clear(comp, ptype)
      #skip if the reader doesn't save the block

    with sharedmem.TPool(np=np) as pool:
      pool.map(work, list(range(Nfile)))

    if skipped_comps:
      warnings.warn('warning: blocks not supported in snapshot: %s', str(skipped_comps))

  def take_snapshots(self, snapshots, ptype, origin=None, boxsize=None, np=None):
    """ ptype can be a list of ptypes, in which case all particles of the types are loaded into the field """
    if numpy.isscalar(ptype):
       ptypes = [ptype]
    else:
       ptypes = ptype

    ptype = None

    nptypes = len(snapshots[0].C['N'])
    N = numpy.zeros((len(snapshots), nptypes), dtype='i8')
    O = N.copy()

    with sharedmem.TPool(np=np) as pool:
      def work(i):
        snapshot = snapshots[i]
        for ptype in ptypes:
          mask, count = filter(snapshot, ptype, origin, boxsize)
          N[i, ptype] = count
        snapshot.clear()
      pool.map(work, range(len(snapshots)))

    O.flat[1:] = N.cumsum()[:-1]
    O = O.reshape(*N.shape)
    self.numpoints = N.sum()
    for comp in self.names:
      shape = list(self[comp].shape)
      shape[0] = self.numpoints
      self[comp] = numpy.zeros(shape, self[comp].dtype)

    with sharedmem.TPool(np=np) as pool:
      def work(i):
        snapshot = snapshots[i]
        for ptype in ptypes:
          mask, count = filter(snapshot, ptype, origin, boxsize)
          for block in snapshot.schema:
            if N[i, ptype] == 0: continue
            if (ptype, block) not in snapshot: continue
            if block not in self.names: continue
            data = select(snapshot, ptype, block, mask)
            self[block][O[i, ptype]:O[i, ptype]+N[i, ptype]] = data
        snapshot.clear()
  
      pool.map(work, range(len(snapshots)))

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
      if index in self.dict:
        return self.dict[index]
      elif index == 'locations':
        return self.dict[self.locations]
      elif index == 'x':
        return self['locations'][:, 0]
      elif index == 'y':
        return self['locations'][:, 1]
      elif index == 'z':
        return self['locations'][:, 2]
      else:
        raise KeyError('index %s not found' % index)
    elif isinstance(index, slice):
      subfield = Field()
      start, stop, step = index.indices(self.numpoints)
      subfield.numpoints = int((stop + step - 1 - start) / step)
      for comp in self.names:
        subfield.dict[comp] = self.dict[comp][index]
      return subfield
    elif isinstance(index, numpy.ndarray) \
       and index.dtype == numpy.dtype('?'):
        subfield = Field()
        subfield.numpoints = int(index.sum())
        for comp in self.names:
          subfield[comp] = self[comp][index]
        return subfield
    elif not numpy.isscalar(index):
      subfield = Field()
      subfield.numpoints = int(len(index))
      for comp in self.names:
        subfield[comp] = self[comp][index]
      return subfield
    else:
      result = {}
      for comp in self.names:
        result[comp] = self[comp][index]
      return result
 
  def __setitem__(self, index, value):
    if index == "locations": index = self.locations
    if isinstance(index, basestring):
      if is_scalar_like(value):
        value = numpy.repeat(value, self.numpoints)
      if value.shape[0] != self.numpoints:
        raise ValueError("num of points of value doesn't match, %d != %d(new)" %( value.shape[0], self.numpoints))
      self.dict[index] = value
      return
    if isinstance(value, dict):
      for comp in value:
        if comp in self:
          self[comp][index] = value[comp]
        else:
          raise IndexError("component %s not in the field" % comp)
      return

    if isinstance(index, slice):
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

  @property
  def dtype(self):
    dtype = []
    for n in self.names:
      if len(self[n].shape) > 1:
        dtype += [(n,  (self[n].dtype, self[n].shape[1:]))]
      else:
        dtype += [(n,  self[n].dtype)]
    return numpy.dtype(dtype)

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

  def smooth(self, tree):
    """ smooth a field. 
        dirty and quick way, estimating sml from the size of the immediate
        tree node containing the particle.
    """
    size = numpy.concatenate((tree['size'][tree._:][:, 0], [0,]))
    npar = tree['npar'][tree._:]
    npar = numpy.concatenate((npar, numpy.array([self.numpoints - npar.sum(),], dtype=npar.dtype)))
    self['sml'] = numpy.repeat(size, npar)
  
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

  def ztree(self, zkey, scale, minthresh, maxthresh):
    from compiledbase import ztree as zt
    return zt.Tree(zkey=zkey, scale=scale, minthresh=minthresh, maxthresh=maxthresh)
    
  def zorder(self, scale=None):
    """ calculate zkey (morton key) and return it.
        if sort is true, the field is sorted by zkey
        if ztree is false, return zkey, scale, where scale is
        an object converting from zkey and coordinates by fillingcurve

        Note all previous reference to the field's components are invalid.
    """
    from compiledbase import fillingcurve as fc
    if scale is None:
      scale = fc.scale(self['locations'].min(axis=0), self['locations'].ptp(axis=0))
    zkey = numpy.empty(self.numpoints, dtype=fc.fckeytype)

    with sharedmem.TPool() as pool:
      chunksize = 1024 * 1024
      def work(i):
        X, Y, Z = self['locations'][i:i+chunksize].T
        fc.encode(X, Y, Z, scale=scale, out=zkey[i:i+chunksize])
      pool.map(work, range(0, len(zkey), chunksize))

    # if already sorted, return directly without copying.
    if (zkey[1:] > zkey[:-1]).all(): return zkey, scale

    # use sharemem.argsort, because it is faster
    arg = sharedmem.argsort(zkey)
    for comp in self.dict:
      # use sharemem.take, because it is faster
      self.dict[comp] = sharedmem.take(self.dict[comp], arg, axis=0)

    zkey = zkey[arg]
    return zkey, scale
