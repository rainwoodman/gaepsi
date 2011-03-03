from readers import Readers
def is_string_like(v):
  try: v + ''
  except: return False
  return True
class Snapshot:
  def __init__(self, file=None, reader=None, *args, **kwargs):
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
      reader = Readers[reader]

    reader.prepare(self, file = file, *args, **kwargs)

  def load(self, blocknames, ptype='all') :
    if hasattr(blocknames, 'isalnum') : blocknames = [blocknames]
    for bn in blocknames: 
      self.reader.load(self, bn, ptype)
  def save(self, blocknames, ptype='all') :
    if hasattr(blocknames, 'isalnum') : blocknames = [blocknames]
    for bn in blocknames: 
      self.reader.save(self, bn, ptype)
 
  def clear(self, blocknames, ptype='all') :
    if hasattr(blocknames, 'isalnum') : blocknames = [blocknames]
    if ptype == None: ptype = 'all'
    for name in blocknames :
      if self.P[ptype].has_key(name): del self.P[ptype][name]

  def __getitem__(self, key) :
    if key == None: key = 'all'
    return self.P[key]
    
