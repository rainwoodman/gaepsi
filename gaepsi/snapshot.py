from gaepsi.readers import get_reader
import numpy

class SnapshotList:
  def __init__(self, all):
    self.all = all
    self.C = all[0].C

  def setN(self, N):
    for i, s in enumerate(self.all):
      start = i * N // len(self.all)
      end = (i + 1)* N // len(self.all)
      s.C['N'][:] = end[:] - start[:]
      s.C['Ntot'][:] = N

  def setC(self, name, value):
    for s in self.all:
      s.C[name] = value

  def __getitem__(self, index):
    ptype, block = index
    return numpy.concatenate([s.load(block, ptype) for s in self.all], axis=0)

  def __setitem__(self, index, value):
    ptype, block = index
    N = 0
    for s in self.all:
      s.P[ptype][block] = value[N:N+s.C['N'][ptype]]
      N = N + s.C['N'][ptype]

  def __contains__(self, index):
    ptype, block = index
    return numpy.any([s.has(block, ptype) for s in self.all])

  def __delitem__(self, index):
    ptype, block = index
    return [s.clear(block, ptype) for s in self.all]

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

  def load(self, name, ptype) :
    """ Load blocks into memory if they are not """
    if name in self.P[ptype]: return self.P[ptype][name]
    if not self.has(name, ptype): return None
    self.reader.load(self, ptype, name)
    return self.P[ptype][name]

  def alloc(self, name, ptype) :
    """ Load blocks into memory if they are not """
    if name in self.P[ptype]: return self.P[ptype][name]
    if not self.has(name, ptype): return None
    self.reader.alloc(self, ptype, name)
    #print 'alloced snap', name, ptype, self.P[ptype][name]
    return self.P[ptype][name]

  def has(self, name, ptype):
    return self.reader.has_block(self, name, ptype)
    
  def save_header(self):
    self.reader.write_header(self)

  def create_structure(self):
    self.reader.create_structure(self)

  def save_all(self):
    self.save_on_delete = False
    self.create_structure()

    # now ensure the structure of the file is complete
    for ptype in range(len(self.C['N'])):
      for name in self.reader.schema:
        if name in self.P[ptype]:
          self.save(name, ptype)

  def save(self, name, ptype, clear=False) :
    self.save_on_delete = False
    self.reader.save(self, ptype, name)
    if clear: 
      self.clear(name, ptype)

  def check(self):
    self.reader.check(self)

  def clear(self, name, ptype) :
    """ relase memory used by the blocks, do not flush to the disk """
    if name in self.P[ptype]: del self.P[ptype][name]

  def __getitem__(self, index):
    ptype, block = index
    return self.load(block, ptype)

  def __setitem__(self, index, value):
    ptype, block = index
    self.P[ptype][block] = value

  def __contains__(self, index):
    ptype, block = index
    return self.has(block, ptype)

  def __delitem__(self, index):
    ptype, block = index
    return self.clear(block, ptype)

