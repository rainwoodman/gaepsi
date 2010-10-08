from numpy import isscalar
from numpy import ones
from matplotlib import is_string_like
from quadtree import QuadTree

class Field:
  """
A field has three elements:
  location, default, and sml(smoothing length)
  """
  def __init__(self, locations=None, sml=None, value=None, snap=None, ptype=0):
    self.dict = {}
    self.snap = snap
    self.ptype = ptype

    boxsize = None
    numpoints = 0
   
    # using a snapshot
    if snap != None:
      snap.push()
      snap.load('pos')
      locations = snap.P[ptype]['pos']
# shall use the schema to determine if a sml is in the snap
      if ptype == 0 :
        snap.load('sml')
        sml = snap.P[ptype]['sml']
      if is_string_like(value):
        snap.load(value)
        value = snap.P[ptype][value]
      boxsize = snap.header['boxsize']
      snap.pop()

    # using given fields
    if boxsize == None:
      boxsize = max(locations[:])

    numpoints = locations.shape[0]
    if value != None:
      if isscalar(value):
        print "scalar value"
        value = ones(numpoints) * value
      self.dict['default'] = value
    if sml != None:
      if isscalar(sml):
        sml = ones(numpoints) * sml
      self.dict['sml'] = sml

    self.numpoints = numpoints
    self.boxsize = boxsize
    self.quadtree = None
    self.dict['locations'] = locations


  def __str__(self) :
    return str(self.dict)

  def __getitem__(self, index):
    if is_string_like(index):
      if self.dict.has_key(index):
        return self.dict[index]
      else:
        if self.snap != None:
          self.snap.push()
          self.snap.load(index)
          self.snap.pop()
          self[index] = self.snap.P[self.ptype][index]
          return self.dict[index]
      raise KeyError("field doesn't exist")
    return ones(self.numpoints) * index

  def __setitem__(self, index, value):
    if is_string_like(index):
      if isscalar(value):
        value = ones(self.numpoints) * value
      if value.shape[0] != self.numpoints:
        raise ValueError("numpoints doesn't match")
      self.dict[index] = value
    else :
      raise KeyError("invalid field name")

  def __delitem__(self, index):
    del self.dict[index]    

  def __contains__(self, index):
    if not isscalar(index): 
      return False
    if is_string_like(index):
      return self.dict.has_key(index)
    return True

  def ensure_quadtree(self):
    if self.quadtree == None:
      pos = self['locations']
      S = self['sml']
      self.quadtree = QuadTree(pos, S, self.boxsize)
  
