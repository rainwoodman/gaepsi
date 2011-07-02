from numpy import isscalar
from numpy import ones,zeros
from numpy import append
from numpy import asarray
from remap import remap
from ccode import NDTree
from ccode import sml
from numpy import sin,cos, matrix
from numpy import inner
from numpy import newaxis

from cosmology import Cosmology

def is_string_like(v):
  try: v + ''
  except: return False
  return True

class Cut:
  def __init__(self, xcut=None, ycut=None, zcut=None, center=None, size=None):
    if xcut != None:
      self.xcut = asarray(xcut)
      self.ycut = asarray(ycut)
      self.zcut = asarray(zcut)
      self.center = zeros(3)
      self.size = zeros(3)
      for cut,axis in [(xcut, 0),(ycut,1),(zcut,2)]:
        self.center[axis] = (cut[0] + cut[1]) / 2
        self.size[axis] = (cut[1] - cut[0])
      return
    if center != None:
      self.center = center[0:3]
      if isscalar(size): size = ones(3) * size
      self.size = size[0:3]
      self.xcut = asarray([center[0] - size[0] / 2, center[0] + size[0] / 2])
      self.ycut = asarray([center[1] - size[1] / 2, center[1] + size[1] / 2])
      self.zcut = asarray([center[2] - size[2] / 2, center[2] + size[2] / 2])
      return
    # uninitialized cut
    self.center = None

  def __repr__(self):
    if self.center == None:
      return 'Cut()'
    return 'Cut(center=%s, size=%s)' % (repr(self.center), repr(self.size))

  def __getitem__(self, index):
    if index == 'x' or index == 0:
      return self.xcut
    if index == 'y' or index == 1:
      return self.ycut
    if index == 'z' or index == 2:
      return self.zcut
  def select(self, locations):
    """return a mask of the locations in the cut"""
    mask = ones(dtype='?', shape = locations.shape[0])
    if self.center == None:
      return mask
    for axis in range(3):
      mask[:] &= (locations[:, axis] >= self[axis][0])
      mask[:] &= (locations[:, axis] < self[axis][1])
    return mask

class Field:
  def __init__(self, components=None, numpoints = 0, boxsize=None, cut=None):
    """components is a dictionary of {component=>dtype}"""
    self.dict = {}
    self.cut = None
    if boxsize != None:
      if isscalar(boxsize):
        self.boxsize = ones(3) * boxsize
      else:
        self.boxsize = boxsize
      self._cut_from_boxsize()
    else:
      self.boxsize = None
    if self.cut == None: self.cut = cut
    self.numpoints = numpoints
    self['locations'] = zeros(shape = numpoints, dtype = ('f4', 3))
    if components != None:
      for comp in components:
        self[comp] = zeros(shape = numpoints, dtype = components[comp])
    self.mask = None
  def set_mask(self, mask):
    if mask != None:
      assert(self.numpoints == mask.size)
    self.mask = mask

  def _cut_from_boxsize(self):
    self.cut = Cut(xcut = [0, self.boxsize[0]],
                   ycut = [0, self.boxsize[1]],
                   zcut = [0, self.boxsize[2]])
    
  def init_from_snapshot(self, snapshot):
    self.boxsize = ones(3) * snapshot.C['L']
    self.cosmology = Cosmology(0, snapshot.C['OmegaM'], snapshot.C['OmegaL'], snapshot.C['h'])
    if self.cut == None:
      self._cut_from_boxsize()

  def add_snapshot(self, snapshot, ptype, components):
    """ components is a dict {component => block}, or a list [ block ]."""
    if self.boxsize == None: 
      self.boxsize = ones(3) * snapshot.C['L']
    if self.cut == None:
      self._cut_from_boxsize()
    snapshot.load(ptype = ptype, blocknames = ['pos'])
    if snapshot.C['N'][ptype] == 0: return 0
    mask = self.cut.select(snapshot.P[ptype]['pos'])
    add_points = mask.sum()
    if add_points == 0: return 0
    self.numpoints = self.numpoints + add_points
    self['locations'] = append(self['locations'], snapshot.P[ptype]['pos'][mask], axis=0)
    snapshot.clear('pos')
    for comp in components:
      try:
        block = components[comp]
      except TypeError:
        block = comp
      snapshot.load(ptype = ptype, blocknames = [block])
      self.dict[comp] = append(self[comp], snapshot.P[ptype][block][mask], axis = 0)
      snapshot.clear(block)
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
      raise ValueError("num of points of value doesn't match")
    self.dict[index] = value
  
  def __delitem__(self, index):
    del self.dict[index]

  def __contains__(self, index):
    return index in self.dict

  def __repr__(self):
    d = {}
    for key in self.dict:
      d[key] = self.dict[key].dtype
    return 'Field(numpoints=%d, components=%s, boxsize=%s, cut=%s>' % (self.numpoints, 
            repr(d), repr(self.boxsize), repr(self.cut))

  def describe(self, index):
    if self.numpoints > 0:
      v = self[index]
      return dict(min=v.min(axis=0), max=v.max(axis=0))
    else:
      return dict(min=None, max=None)

  def dist(self, center):
    d2 = ((self['locations'] - center) ** 2).sum(axis=1)
    return d2 ** 0.5

  def smooth(self, NGB=32):
    self['sml'] = sml(locations = self['locations'], mass = self['mass'], N=NGB)

  def rotate(self, angle, axis, center):
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

    self['locations'] -= center
    self['locations'] = inner(self['locations'], M)
    self['locations'] += center
    for comp in self.dict.keys():
      if comp != 'locations':
        if len(self[comp].shape) > 1:
          self[comp] = inner(self[comp], M)

  def unfold(self, M):
    """ unfold the field position by transformation M
        the field shall be periodic. M is an
        list of column integer vectors of the shearing
        vectors. abs(det(M)) = 1
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
