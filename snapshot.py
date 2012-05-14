from gaepsi.readers import get_reader

class Snapshot:
  def __init__(self, file=None, reader=None, create=False, overwrite=True, **kwargs):
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

    # block offset table
    self.sizes = {}
    self.offsets = {}

    if reader == None: return

    reader = get_reader(reader)

    self.file = file
    if create:
      self.save_on_delete = True
      reader.create(self, overwrite=overwrite)
      for key in kwargs:
        self.header[key] = kwargs[key]

    else:
      self.save_on_delete = False
      reader.open(self)

  def __del__(self):
    if hasattr(self, 'save_on_delete') and self.save_on_delete:
#      print 'saving snapshot %s at destruction' % self.file
      self.save_all()

  def load(self, blocknames, ptype) :
    """ Load blocks into memory if they are not """
    if hasattr(blocknames, 'isalnum') : blocknames = [blocknames]
    for bn in blocknames:
      if ptype in self.P and bn in self.P[ptype]: continue
      if not self.has(bn, ptype): continue
      self.reader.load(self, ptype, bn)
    print ptype, bn
    return [self.P[ptype][bn] for bn in blocknames]

  def has(self, blockname, ptype):
    return self.reader.has_block(self, ptype, blockname)
    
  def create_structure(self):
    self.reader.create_structure(self)

  def save_all(self):
    self.save_on_delete = False
    self.reader.create_structure(self)

    # now ensure the structure of the file is complete
    for ptype in range(6):
      for block in [sch['name'] for sch in self.reader.schemas]:
        if block in self.P[ptype]:
          self.save(ptype = ptype, blocknames = [block])
# no need to flush as the file is supposingly closed.
#    self.file.flush()

  def save(self, blocknames, ptype) :
    self.save_on_delete = False
    if hasattr(blocknames, 'isalnum') : blocknames = [blocknames]
    for bn in blocknames: 
      self.reader.save(self, ptype, bn)
 
  def check(self):
    self.reader.check(self)

  def clear(self, blocknames, ptype) :
    """ relase memory used by the blocks, do not flush to the disk """
    if hasattr(blocknames, 'isalnum') : blocknames = [blocknames]
    for name in blocknames :
      if self.P[ptype].has_key(name): del self.P[ptype][name]

  def __getitem__(self, index):
    ptype, block = index
    return self.load([block], ptype)[0]

  def __contains__(self, index):
    ptype, block = index
    return self.has(block, ptype)

  def __delitem__(self, index):
    ptype, block = index
    return self.clear([block], ptype)

