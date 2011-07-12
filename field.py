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

import threading

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
      self.size = size[0:3]
      return
    # uninitialized cut
    self.center = None

  def take(self, cut):
    if cut is not None:
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
    mask = ones(dtype='?', shape = locations.shape[0])
    if self.center == None:
      return mask
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
    if components != None:
      for comp in components:
        self[comp] = zeros(shape = numpoints, dtype = components[comp])
    self.mask = None

    self.__lock = threading.RLock()

  @property
  def boxsize(self):
    return self.cut.size
  @boxsize.setter
  def boxsize(self, value):
    if isscalar(value): value = ones(3) * value
    for axis in range(3):
      self.cut[axis] = (0, value[axis])
  
  def set_mask(self, mask):
    if mask != None:
      assert(self.numpoints == mask.size)
    self.mask = mask

  def init_from_snapshot(self, snapshot, cut=None):
    if cut is None:
      self.boxsize = snapshot.C['L']
    else: self.cut.take(cut)

    self.cosmology = Cosmology(0, snapshot.C['OmegaM'], snapshot.C['OmegaL'], snapshot.C['h'])

  def comp_to_block(self, comp):
    if comp == 'locations': return 'pos'
    return comp

  def add_snapshot(self, snapshot, ptype):
    """ """

    if snapshot.C['N'][ptype] == 0: return 0

    for comp in self:
      snapshot.load(ptype = ptype, blocknames = [self.comp_to_block(comp)])

    mask = self.cut.select(snapshot.P[ptype]['pos'])
    add_points = mask.sum()
    if add_points == 0: return 0

    with self.__lock:
      for comp in self:
        self.dict[comp] = append(self[comp], 
            snapshot.P[ptype][self.comp_to_block(comp)][mask], 
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
