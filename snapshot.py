import numpy
import gadget
class Snapshot:
  def __init__(self, fname, schema):
    if hasattr(schema, 'open'):
      self.schema = schema
    else:
      self.schema = gadget.schemas[schema]

    # constants (cosmology and stuff)
    self.C = {}
    # particle data
    self.P = [{}, {}, {}, {}, {}, {}]
    self.__prevP = None
    # block data
    self.D = {}
    self.__prevD = None
    # block offset table
    self.sizes = {}
    self.offsets = {}

    self.file = None
    self.schema.open(self, fname)

    self.header = self.file.read_record(self.schema.header, 1)[0]
    self.Nparticle = None

    self.schema.update_meta(self, self.header)
    self.schema.update_offsets(self)
    self.schema.post_init(self)

  def push(self) :
    "pushing the list of currently loaded fields"
    self.__prevD = self.D
    self.__prevP = self.P
  def pop(self) :
    "popping the list of currently loaded fields"
    if(self.__prevD != None):
      self.D = self.__prevD
      self.__prevD = None
      self.P = self.__prevP
      self.__prevP = None

  def load(self, blocknames) :
    if hasattr(blocknames, 'isalnum') : blocknames = [blocknames]
    for name in blocknames :
      # if already loaded, don't reload
      if self.D.has_key(name) : continue

      if not self.schema.has_key(name): 
        raise KeyError("block not found in snapshot file")

      self.file.seek(self.offsets[name])
      blockschema = self.schema.get(name)
      if self.sizes[name] != 0 :
        length = self.sizes[name] // blockschema.dtype.itemsize
        self.D[name] = self.file.read_record(blockschema.dtype, length)
      else :
        self.D[name] = None
      self.schema.reindex(self, name)
  def clear(self, blocknames) :
    if hasattr(blocknames, 'isalnum') : blocknames = [blocknames]
    for name in blocknames :
      if self.D.has_key(name) :
        del self.D[name]
      for P in self.P:
        if P.has_key(name): del P[name]

