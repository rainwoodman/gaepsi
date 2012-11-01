from gaepsi.readers import get_reader

class Snapshot:
  def __init__(self, file=None, reader=None, create=False, overwrite=True, **kwargs):
    """ creats a snapshot
      **kwargs are the fields in the header to be filled if create=True.
      **kwargs are ignored when create=False.
    """
    # block offset table
    self.sizes = {}
    self.offsets = {}

    reader = get_reader(reader)

    if reader == None: 
      raise Excpetion('reader %s is not found' % reader)


    self.file = file
    if create:
      self.save_on_delete = True
      reader.create(self, overwrite=overwrite)
      for key in kwargs:
        self.C[key] = kwargs[key]

    else:
      self.save_on_delete = False
      reader.open(self)

    #self.C is set after reader.create / reader.open
    # particle data
    self.P = {}
    for n in range(len(self.C['N'])):
      self.P[n] = {}
    self.P[None] = {}

  def __del__(self):
    if hasattr(self, 'save_on_delete') and self.save_on_delete:
#      print 'saving snapshot %s at destruction' % self.file
      self.save_all()

  def load(self, blocknames, ptype) :
    """ Load blocks into memory if they are not """
    if hasattr(blocknames, 'isalnum') : blocknames = [blocknames]
    for bn in blocknames:
      if bn in self.P[ptype]: continue
      if not self.has(bn, ptype): continue
      self.reader.load(self, ptype, bn)
    return [self.P[ptype][bn] for bn in blocknames]

  def has(self, blockname, ptype):
    return self.reader.has_block(self, blockname, ptype)
    
  def save_header(self):
    self.reader.write_header(self)

  def create_structure(self):
    self.reader.create_structure(self)

  def save_all(self):
    self.save_on_delete = False
    self.create_structure()

    # now ensure the structure of the file is complete
    for ptype in range(len(self.C['N'])):
      for block in self.reader.schema:
        if block in self.P[ptype]:
          self.save(ptype=ptype, blocknames=[block])

  def save(self, blocknames, ptype, clear=False) :
    self.save_on_delete = False
    if isinstance(blocknames, basestring) : 
      blocknames = [blocknames]
    for bn in blocknames: 
      self.reader.save(self, ptype, bn)
    if clear: 
      self.clear(blocknames, ptype)

  def check(self):
    self.reader.check(self)

  def clear(self, blocknames, ptype) :
    """ relase memory used by the blocks, do not flush to the disk """
    if hasattr(blocknames, 'isalnum') : blocknames = [blocknames]
    for name in blocknames :
      if name in self.P[ptype]: del self.P[ptype][name]

  def __getitem__(self, index):
    ptype, block = index
    return self.load([block], ptype)[0]

  def __setitem__(self, index, value):
    ptype, block = index
    self.P[ptype][block] = value

  def __contains__(self, index):
    ptype, block = index
    return self.has(block, ptype)

  def __delitem__(self, index):
    ptype, block = index
    return self.clear([block], ptype)

