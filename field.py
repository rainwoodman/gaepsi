from numpy import isscalar
from numpy import ones,zeros
from numpy import append
from remap import remap
from ccode import NDTree

def is_string_like(v):
  try: v + ''
  except: return False
  return True

class Field:
  def __init__(self, components=None, numpoints = 0, origin = [0., 0., 0.], boxsize=None, xcut=None, ycut=None, zcut=None):
    """components is a dictionary of {component=>dtype}"""
    if origin != None:
      self.origin = origin
    self.dict = {}
    if boxsize != None:
      if isscalar(boxsize):
        self.boxsize = ones(3) * boxsize
      else:
        self.boxsize = boxsize
    self.xcut = xcut
    self.ycut = ycut
    self.zcut = zcut
    self.numpoints = numpoints
    self['locations'] = zeros(shape = numpoints, dtype = ('f4', 3))
    if components != None:
      for comp in components:
        self[comp] = zeros(shape = numpoints, dtype = components[comp])

  def add_snapshot(self, snapshot, ptype, components):
    """ components is a dict {component => block}, or a list [ block ]."""
    if self.boxsize == None: 
      self.boxsize = ones(3) * snapshot.C['L']
    snapshot.load(ptype = ptype, blocknames = ['pos'])
    locations = snapshot.P[ptype]['pos']
    mask = ones(dtype='?', shape = snapshot.P[ptype]['pos'].shape[0])

    for cut, axis in [(self.xcut, 0), (self.ycut, 1), (self.zcut, 2)]:
      if cut != None:
        mask[:] &= (locations[:, axis] >= cut[0])
        mask[:] &= (locations[:, axis] < cut[1])

    add_points = mask.sum()
    self.numpoints = self.numpoints + add_points
    temp = locations[mask]
    self['locations'] = append(self['locations'], temp, axis=0)
    del locations
    snapshot.clear('pos')
    for comp in components:
      try:
        block = components[comp]
      except TypeError:
        block = comp
      snapshot.load(ptype = ptype, blocknames = [block])
      self.dict[comp] = append(self[comp], snapshot.P[ptype][block][mask], axis = 0)
      snapshot.clear(block)

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

  def describe(self, index):
    if self.numpoints > 0:
      v = self[index]
      return dict(min=v.min(axis=0), max=v.max(axis=0))
    else:
      return dict(min=None, max=None)

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
