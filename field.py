from numpy import isscalar
from numpy import ones,zeros
from matplotlib import is_string_like
from quadtree import QuadTree
from octtree import OctTree
from remap import remap

class Field:
  """
A field has three elements:
  location, default, and sml(smoothing length)
  """
  def __init__(self, snap=None, ptype=None, locations=None, sml=None, periodical = True, value=None):
    self.dict = {}
    self.snap = snap
    self.ptype = ptype
    self.periodical = periodical
    boxsize = None
    origin = None
    numpoints = 0
   
    # using a snapshot
    if snap != None:
      if ptype == None: ptype = 'all'
      if locations == None:
        snap.load('pos', ptype)
        locations = snap.P[ptype]['pos']
        origin = zeros(3)
        boxsize = ones(3) * snap.C['L']
# shall use the schema to determine if a sml is in the snap
      if sml == None and ptype == 0 :
        snap.load('sml', 0)
        sml = snap.P[0]['sml']
      if is_string_like(value):
        snap.load(value, ptype)
        value = snap.P[ptype][value]

    # boxsize using given fields
    if origin == None:
      origin = locations.min(axis=0)
    if boxsize == None:
      boxsize = locations.max(axis=0) - origin

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
    self.origin = origin
    self.quadtree = None
    self.octtree = None
    self.dict['locations'] = locations


  def __str__(self) :
    return str(self.dict)

  def __getitem__(self, index):
    if is_string_like(index):
      if self.dict.has_key(index):
        return self.dict[index]
      else:
        if self.snap != None:
          self.snap.load(index, self.ptype)
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
      self.quadtree = QuadTree(pos, S, self.origin, self.boxsize, self.periodical)

  def ensure_octtree(self):
    if self.octtree == None:
      pos = self['locations']
      S = self['sml']
      self.octtree = OctTree(pos, S, self.origin, self.boxsize, self.periodical)
      
  def unfold(self, M):
    """ unfold the field position by transformation M
        the field shall be periodic. M is an
        list of column integer vectors of the shearing
        vectors. abs(det(M)) = 1
    """
    if self.periodical == False: 
      raise ValueError("the field must be periodic")
    boxsize = self.boxsize[0]
    for b in self.boxsize:
      if b != boxsize:
        raise ValueError("the box has to be cubic.")

    pos = self['locations']
    pos /= boxsize
    newpos,QT,newboxsize, badmask = remap(M, pos)
    if badmask.any(): 
      raise ValueError("failed to remap some points")
    newpos *= boxsize
    self['locations'] = newpos
    self.boxsize = newboxsize * boxsize
    
     
