
def is_string_like(v):
  try: v + ''
  except: return False
  return True
class Snapshot:
  def __init__(self, file=None, reader=None, create=False, **kwargs):
    """ creats a snapshot
      **kwargs are the fields in the header to be filled if create=True.
      **kwargs are ignored when create=False.
    """
    # constants (cosmology and stuff)
    self.C = None
    # particle data
    self.P = {}
    for i in range(6):
      self.P[i] = {}
    self.P['all'] = {}

    # block offset table
    self.sizes = {}
    self.offsets = {}

    if reader == None: return

    if is_string_like(reader) :
      _temp = __import__('gaepsi.readers.%s' % reader, globals(), locals(),
              ['Reader'],  -1)
      reader = _temp.Reader()

    if create:
      self.save_on_delete = True
      reader.create(self, file = file)
      for key in kwargs:
        self.header[key] = kwargs[key]

    else:
      self.save_on_delete = False
      reader.open(self, file = file)

  def __del__(self):
    if self.save_on_delete:
      print 'saving snapshot %s at destruction' % self.file.name
      self.save_all()

  def load(self, blocknames, ptype='all') :
    if hasattr(blocknames, 'isalnum') : blocknames = [blocknames]
    for bn in blocknames: 
      self.reader.load(self, bn, ptype)

  def save_all(self):
    self.save_on_delete = False
    self.save(blocknames = 'header')
    print self.offsets
    print self.sizes
    for ptype in range(6):
      for block in [sch['name'] for sch in self.reader.schemas]:
        if block in self.P[ptype]:
          self.save(ptype = ptype, blocknames = [block])

  def save(self, blocknames, ptype='all') :
    self.save_on_delete = False
    if hasattr(blocknames, 'isalnum') : blocknames = [blocknames]
    for bn in blocknames: 
      self.reader.save(self, bn, ptype)
 
  def check(self):
    self.reader.check(self)

  def clear(self, blocknames, ptype='all') :
    if hasattr(blocknames, 'isalnum') : blocknames = [blocknames]
    if ptype == None: ptype = 'all'
    for name in blocknames :
      if self.P[ptype].has_key(name): del self.P[ptype][name]

  def __getitem__(self, key) :
    if key == None: key = 'all'
    return self.P[key]
    
