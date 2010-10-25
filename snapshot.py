import gadget
from matplotlib import is_string_like
    
class Snapshot:
  def __init__(self, file=None, reader=None):
    # constants (cosmology and stuff)
    self.C = {}
    # particle data
    self.P = {}
    for i in range(6):
      self.P[i] = {}
    self.P['all'] = {}

    # block offset table
    self.sizes = {}
    self.offsets = {}

    self.N = None
    if reader == None: return

    if is_string_like(reader) :
      reader = gadget.Readers[reader]

    reader.prepare(self, file)

  def load(self, blocknames, ptype=None) :
    if hasattr(blocknames, 'isalnum') : blocknames = [blocknames]
    for bn in blocknames: 
      self.reader.load(self, bn, ptype)
 
  def clear(self, blocknames, ptype=None) :
    if hasattr(blocknames, 'isalnum') : blocknames = [blocknames]
    if ptype == None: ptype = 'all'
    for name in blocknames :
      if self.P[ptype].has_key(name): del P[name]

  def __getitem__(self, key) :
    if key == None: key = 'all'
    return self.P[key]
    
