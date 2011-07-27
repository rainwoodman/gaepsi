from ccode import OctTree as _OctTree

class OctTree:
  def __init__(self, field):
    self.field = field
    self._tree = _OctTree(locations = field['locations'], sml=field['sml'], boxsize=field.cut.size, origin=field.cut.origin)
  def __repr__(self):
    return "OctTree: occupied %d/%d" %(self._tree.pool_length, self._tree.pool_size)

  def __getitem__(self, index):
    return self._tree.get_cell(index)

  def trace(self, src, dir, dist=None):
    if dist is None:
      dist = self._tree.boxsize[0] * 2
    return self._tree.trace(src=src, dir=dir, dist=dist)
  
