from gaepsi.readers import Reader
from gaepsi.io import BlockSizeError
import numpy

class Snapshot:
  def __init__(self, file=None, reader=None, 
               create=False, overwrite=True, **kwargs):
    reader = Reader(reader, **kwargs)
    self.reader = reader
    self.file = file
    if create:
      self.save_on_delete = True
      self.create(overwrite=overwrite)
    else:
      self.save_on_delete = False
      self.open()

    #self.C is set after reader.create / reader.open
    # particle data
    self.P = {}
    for n in range(len(self.C['N'])):
      self.P[n] = {}
    self.P[None] = {}

  def getfile(self, mode):
    return self.reader.file_class(self.file, endian=self.reader.endian, mode=mode)
  def open(self):
    with self.getfile('r') as file:
      fileheader = file.read_record(self.reader.header, 1).squeeze()
      self.C = self.reader.constants(fileheader, init=False, extra=self.reader.extra_kwargs)
      self.schema = Schema(self.reader, self.C)
    self.resolve_schema()

  def create(self, overwrite):
    if not overwrite:
      mode = 'wx+'
    else:
      mode = 'w+'
    with self.getfile(mode) as file:
      self.C = self.create_header()
      self.schema = Schema(self.reader, self.C)

  def create_header(self):
    fileheader = numpy.zeros(dtype=self.reader.header, shape=None)
    return self.reader.constants(fileheader, init=True, extra=self.reader.extra_kwargs)

  def _decodeindex(self, index):
      if isinstance(index, tuple):
        ptype, name = index
      else:
        ptype = None
        name = index 
      return ptype, name
  def __getitem__(self, index):
    """ if ptype is None, read in all ptypes """
    with self.getfile('r') as file:
      ptype, name = self._decodeindex(index)
      dtype = self.schema[name].dtype
      ptypes = self.schema[name].ptypes
      if ptype is None:
        file.seek(self.offset(name))
        length = self.size(name) // dtype.itemsize
        self.P[None][name] = file.read_record(dtype, length)
        for ptype in ptypes:
          if self.needmasstab(name, ptype):
            self.P[ptype][name] = numpy.empty(self.C['N'][ptype], dtype)
            self.P[ptype][name][:] = self.C['mass'][ptype]
          else:
            offset = self.count_particles(name, ptype)
            self.P[ptype][name] = self.P[None][name]\
                  [offset:offset + self.C['N'][ptype]]
      else:
        if not self.has_block(name, ptype):
          raise IOError('block %s does not exist in file %s for ptype %d'\
                     % (name, self.file, ptype))

        if self.needmasstab(name, ptype):
          self.P[ptype][name] = numpy.empty(self.C['N'][ptype], dtype)
          self.P[ptype][name][:] = self.C['mass'][ptype]
        else:
          file.seek(self.offset(name))
          length = self.size(name) // dtype.itemsize
          offset = self.count_particles(name, ptype)
          self.P[ptype][name] = file.read_record(dtype, length, offset, self.C['N'][ptype])
    return self.P[ptype][name]

  def __contains__(self, index):
    ptype, name = self._decodeindex(index)
    return self.has_block(name, ptype=ptype)
  
  def __del__(self):
    if hasattr(self, 'save_on_delete'):
      if self.save_on_delete:
        self.save(None)

  def __delitem__(self, index):
    ptype, name = self._decodeindex(index)
    if name in self.P[ptype]: 
      del self.P[ptype][name]
    if ptype is None:
      ptypes = self.schema[name].ptypes
      for ptype in ptypes:
         del self[ptype, name]
       
  def save(self, index):
    if index is None:
      for ptype in range(len(self.C['N'])):
        for name in self.schema:
          if name in self.P[ptype]:
            if name is not None:
              self.save((ptype, name))
    ptype, name = self._decodeindex(index)
    if not self.has_block(name, ptype):
      return
    if self.needmasstab(name, ptype):
      if (self.P[ptype][name] != self.C['mass'][ptype]).any():
        warnings.warn('mismatching particle mass detected')
      return
    with self.getfile('r+') as file:
      file.seek(self.offset(name))
      dtype = self.schema[name].dtype
      length = self.size(name) // dtype.base.itemsize
      offset = self.count_particles(name, ptype)
      offset *= dtype.itemsize // dtype.base.itemsize
      if len(self.P[ptype][name]) != self.C['N'][ptype]:
        raise IOError('snap memory image corrupted: %s %d %d %d' \
              % (name, ptype, len(self.P[ptype][name]), 
                 self.C['N'][ptype]))
    file.write_record(self.P[ptype][name], length, offset)

  def resolve_schema(self):
    bytesize = [0, 4, 8, 16, 32, -1]
    with self.getfile('r') as file:
      for name in self.schema:
        if not self.has_block(name): continue
        s = self.schema[name]
        for newbs in bytesize:
          if newbs == -1:
            raise Exception("Cannot resolve the schema on this \
            file, do not understand the block sizes")
          if newbs != 0:
            s.modify_dtype(newbs)
          file.seek(self.offset(name))
          length = self.size(name) // s.dtype.itemsize
          try:
            file.skip_record(s.dtype, length)
            break
          except BlockSizeError:
            pass
         
  def check(self):
    with self.getfile('r') as file:
      for name in self.schema:
        s = self.schema[name]
        if not self.has_block(name): continue;
        file.seek(self.offset(name))
        length = self.size(name) // s.dtype.itemsize
        file.skip_record(s.dtype, length)
     
  def offset(self, name):
    offset = self.reader.file_class.get_size(
              self.reader.header.itemsize);
    if not self.has_block(name):
      raise IOError('block %s does not exist in file' % name)
    for iname in self.schema:
      if not self.has_block(iname):
        continue
      if iname == name: break
      N = self.count_particles(iname, None)
      blocksize = N * self.schema[iname].dtype.itemsize

      offset += self.reader.file_class.get_size(blocksize);
    return offset

  def size(self, name):
    if not self.has_block(name):
      raise IOError('block %s does not exist in file' % name)
    N = self.count_particles(name, None)
    blocksize = N * self.schema[name].dtype.itemsize
    return blocksize;
    
  def has_block(self, name, ptype=None):
    if not name in self.schema: return False
    schema = self.schema[name]
    for cond in schema.conditions:
        if self.C[cond] == 0 : return False
    if ptype is None: return True
    if ptype in schema.ptypes: return True
    return False

  def count_particles(self, name, endptype=None):
    N = 0
    schema = self.schema[name]
    if endptype is None:
      endptype = len(self.C['N'])
    for ptype in range(endptype):
      if self.needmasstab(name, ptype):
        continue
      if ptype in schema.ptypes:
        N += int(self.C['N'][ptype])
    return N

  def needmasstab(self, name, ptype):
    return name == 'mass' \
       and 'mass' in self.C \
       and self.C['mass'][ptype] != 0.0
    
from collections import OrderedDict, namedtuple

class SchemaEntry(object):
    def __init__(self, name, dtype, ptypes, conditions):
        self.name = name
        self.dtype = numpy.dtype(dtype)
        self.ptypes = numpy.atleast_1d(numpy.array(ptypes,
            dtype='i4'))
        self.conditions = conditions
    def modify_dtype(self, newnbytes):
        self.dtype = numpy.dtype((self.dtype.base.kind +
                str(newnbytes), self.dtype.shape))

class Schema(OrderedDict):
  def __init__(self, reader, header):
    OrderedDict.__init__(self)
    allptypes = numpy.arange(len(header['N']))
    for name in reader.schema.__blocks__:
      entry = getattr(reader.schema, name)
      if len(entry) == 1:
        entry = SchemaEntry(name=name, 
            dtype=entry[0], 
            ptypes=allptypes, 
            conditions=())
      elif len(entry) == 2:
        entry = SchemaEntry(name=name, 
            dtype=entry[0],
            ptypes=entry[1], 
            conditions=())
      elif len(entry) == 3:
        entry = SchemaEntry(name=name, 
            dtype=entry[0], 
            ptypes=entry[1], 
            conditions=entry[2])
      else:
        raise SyntaxError('schema declaration is wrong')
      self[name] = entry

