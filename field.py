from numpy import isscalar
from numpy import ones,zeros
from numpy import append
from numpy import asarray
from remap import remap
from ccode import sml
from numpy import sin,cos, matrix
from numpy import inner
from numpy import newaxis

from cosmology import Cosmology

from tools import threads
from Queue import Queue

def is_string_like(v):
  try: v + ''
  except: return False
  return True

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
      if isscalar(size): size = ones(3) * size
      if center is not None:
        self.center = asarray(center[0:3])
      else:
        self.center = asarray(size) * 0.5
      self.size = asarray(size[0:3])
      return
    # uninitialized cut
    self.center = None

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
    if components is not None:
      for comp in components:
        self.dict[comp] = zeros(shape = numpoints, dtype = components[comp])
    self.mask = None

  @property
  def tree(self):
    if self.__tree__ is None:
      self.__tree__ = OctTree(f)
    return self.__tree__

  @property
  def boxsize(self):
    return self.cut.size
  @boxsize.setter
  def boxsize(self, value):
    if isscalar(value): value = ones(3) * value
    self.cut.center = zeros(3)
    self.cut.size = zeros(3)
    for axis in range(3):
      self.cut[axis] = (0, value[axis])
  
  def set_mask(self, mask):
    if mask != None:
      assert(self.numpoints == mask.size)
    self.mask = mask

  def init_from_snapshot(self, snapshot, cut=None):
    if cut is None:
      self.boxsize = snapshot.C['boxsize']
    else: self.cut.take(cut)

    self.cosmology = Cosmology(0, snapshot.C['OmegaM'], snapshot.C['OmegaL'], snapshot.C['h'])

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
      snapshot.C['OmegaM'] = self.cosmology.Omega['M']
      snapshot.C['OmegaL'] = self.cosmology.Omega['L']
      snapshot.C['h'] = self.cosmology.h
      snapshot.C['boxsize'] = self.boxsize[0]
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
    print 'warning: blocks not supported in snapshot', skipped_comps

  def take_snapshots(self, snapshots, ptype):
    num_workers = 8
    job1_q = Queue()
    job2_q = Queue()

    self.init_from_snapshot(snapshots[0])

    self.numpoints = 0

    @threads.job
    def job1(snapshot, lock):
      snapshot.load(ptype = ptype, blocknames = ['pos'])
      if snapshot.C.N[ptype] != 0:
        mask = self.cut.select(snapshot.P[ptype]['pos'])
        if mask is not None:
          length = mask.sum()
        else:
          length = snapshot.C.N[ptype]
      else:
        length = 0
        mask = None
      with lock:
        start = self.numpoints
        job2 = (mask, snapshot, start, length)
        self.numpoints = self.numpoints + length
      job2_q.put(job2)

    for snapshot in snapshots:
      job1_q.put((snapshot,))

    threads.work(job1, job1_q)

    # allocate the storage space, trashing whatever already there.
    for comp in self:
      shape = list(self[comp].shape)
      shape[0] = self.numpoints
      self.dict[comp] = zeros(shape = shape,
         dtype = self.dict[comp].dtype)

    skipped_comps = set([])

    @threads.job
    def job2(mask, snapshot, start, length, lock):
      if length == 0: 
        return
      for comp in self:
        try:
          snapshot.load(ptype = ptype, blocknames = [self.comp_to_block(comp)])
        except KeyError:
          # skip blocks that are not in the snapshot
          skipped_comps.update(set([comp]))
          continue
        if mask is None:
          self[comp][start:start+length] = snapshot.P[ptype][self.comp_to_block(comp)][:]
        else:
          self[comp][start:start+length] = snapshot.P[ptype][self.comp_to_block(comp)][mask]
        snapshot.clear(self.comp_to_block(comp))

    threads.work(job2, job2_q)

    print 'warning: comp not suppored by the snapshot', skipped_comps

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
    return self.dict[index]

  def __setitem__(self, index, value):
    if isscalar(value):
      value = ones(self.numpoints) * value
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

  def smooth(self, NGB=32):
    self['sml'] = sml(locations = self['locations'], mass = self['mass'], N=NGB)

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

  def redshift_distort(self, dir, redshift, vel=None):
    """ perform redshift distortion along direction dir, needs 'vel' and 'pos'
        if vel is None, field['vel'] is converted to peculiar velocity via multiplying by sqrt(1 / ( 1 + redshift)). A constant H calculated from redshift is used. The position are still given in comoving distance units, NOT the velocity unit. 
    """
    a = 1 / (1. + redshift)
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
    boxsize = self.boxsize[0]
    for b in self.boxsize:
      if b != boxsize:
        raise ValueError("the box has to be cubic.")

    pos = self['locations']
    pos /= boxsize
    newpos,newboxsize = remap(M, pos)
    newpos *= boxsize
    self['locations'] = newpos
    self.boxsize = newboxsize * boxsize
