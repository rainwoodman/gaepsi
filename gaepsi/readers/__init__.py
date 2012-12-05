import numpy
import warnings
from gaepsi.tools import virtarray
from types import MethodType
from gaepsi.io import CFile, F77File

def is_string_like(v):
  try: v + ''
  except: return False
  return True

def get_reader(reader):
  return Reader(reader)

def Reader(reader, forcedouble=False):
  """ first assume reader is a module name and import Reader class from it.
      if failed assume reader is a qualified classname and import 
      the module and class.
  """
  if is_string_like(reader) :
    try:
      module = __import__('gaepsi.readers.%s' % reader, globals(), {}, [''], 0)
      modulename = reader
      classname = 'Snapshot'
    except:
      modulename, classname = reader.rsplit('.', 1)
      module = __import__('gaepsi.readers.%s' % modulename, globals(), {}, [classname], 0)
      
    if not hasattr(module, classname):
      raise ImportError('Reader class %s not found in %s', classname, modulename)
    reader = getattr(module, classname)
  if not isinstance(reader, ReaderObj):
    reader = ReaderObj(reader)
  if forcedouble:
    reader.schema.force_double_precision()

  return reader

class ConstBase:
  """
     a constant can be at the following locations:
     the header, (saved in the snapshot file)
     the class,  (may be converted to something in the header)
     or extra.   (lost)
  """
  def __init__(self, header, init=False):
    """ if init is True, the values are initialized """
    self._header = header
    # extra is the extra values
    self._extra = {}
    if init:
      for item in self._header.dtype.names:
        if hasattr(self, item):
          attr = getattr(self, item)
          if isinstance(attr, (tuple, basestring)): continue
          self._header[item] = attr
      
  def __iter__(self):
    def generator():
      for name in sorted(numpy.unique(dir(self) \
         + list(self._header.dtype.names) \
         + list(self._extra.keys()))):
        if name[0] == '_': continue
        yield name
    return generator()
    
  def __contains__(self, item):
    return item in self._header.dtype.names or hasattr(self, item) or item in self._extra

  def __getitem__(self, item):
    if item in self._header.dtype.names:
      return self._header[item]
    elif hasattr(self, item):
      attr = getattr(self, item)
      if isinstance(attr, basestring):
        return self._header[attr]
      elif isinstance(attr, tuple) or isinstance(attr, list):
        dtype, get, set = attr
        return virtarray(None, dtype, MethodType(get, self, None), MethodType(set, self, None))
      else:
        return attr
    else:
      return self._extra[item]

  def __setitem__(self, item, value):
    if item is Ellipsis:
      assert isinstance(value, ConstBase)
      self._header[...] = value._header
      self._extra = value._extra.copy()
    elif item in self._header.dtype.names:
      self._header[item] = value
    elif hasattr(self, item):
      attr = getattr(self, item)
      if isinstance(attr, basestring):
        self._header[attr] = value
      elif isinstance(attr, tuple) or isinstance(attr, list):
        dtype, get, set = attr
        virtarray(None, dtype, MethodType(get, self, None), MethodType(set, self, None))[...] = value
      else:
        raise IndexError("can't set %s" % item)
    else:
      self._extra[item] = value

  def __str__(self):
    s = []
    for name in self:
      if name[0] == '_': continue
      if name in self._header.dtype.names:
        s += ['%s(header) = %s' % (name, str(self[name]))]
      elif name in self._extra:
        s += ['%s(extra) = %s' % (name, str(self[name]))]
      else:
        s += ['%s(class) = %s' % (name, str(self[name]))]
    return '\n'.join(s)

class Schema:
  from collections import namedtuple
  Entry = namedtuple("Entry", ['name', 'dtype', 'ptypes', 'conditions'])
  def __init__(self, config, nptypes):
    self.__blocks__ = config.__blocks__
    all = range(nptypes)
    for block in self.__blocks__:
      entry = getattr(config, block)
      if len(entry) == 1:
        dtype = entry
        ptypes = all
        conditions = None
      elif len(entry) == 2:
        dtype, ptypes = entry
        conditions = None
      else:
        dtype, ptypes, conditions = entry
      try: ptypes[0]
      except :
        ptypes = (ptypes,)
      dtype = numpy.dtype(dtype)
      self.__dict__[block] = Schema.Entry._make((block, dtype, ptypes, conditions))
    self.nptypes = nptypes

  def __contains__(self, index):
    return index in self.__dict__
  def __getitem__(self, index):
    return self.__dict__[index]
  def __iter__(self):
    return iter(self.__blocks__)
  def __str__(self):
    return str(self.__blocks__)

class ReaderObj(object):
  filedict = {'F': F77File, 'C': CFile }
  properties = {
      'format': ('F', lambda x: x),
      'header': (None, numpy.dtype),
      'schema': (None, lambda x: x),
      'constants': ({}, lambda x: type('constants', (x, ConstBase, object), {})),
      'usemasstab': (True, lambda x: x),
      'endian': ('<', lambda x: x),
    }

  def __init__(cls, config):
    missing = []

    for p in ReaderObj.properties:
      desc = ReaderObj.properties[p]
      value, constructor = desc
      if hasattr(config, p):
        cls.__dict__[p] = constructor(getattr(config, p))
      else:
        if value is not None:
          cls.__dict__[p] = constructor(value)
        else:
          missing.append(p)

    if missing:
      raise ValueError("missing class properties in %s: %s" %
       (config.__name__, ', '.join(missing)))
    cls.file_class = cls.filedict[cls.format]

    if hasattr(cls.constants, 'N'):
      nptypes = numpy.dtype(cls.constants.N[0]).shape[0]
    else:
      nptypes = cls.header['N'].shape[0]
    cls.schema = Schema(cls.schema, nptypes)

  def __str__(cls):
    return str(cls.__dict__)

  def open(cls, snapshot):
    file = cls.file_class(snapshot.file, endian=cls.endian, mode='r')
    snapshot.reader = cls
    file.seek(0)
    snapshot.header = file.read_record(cls.header, 1).squeeze()
    snapshot.C = cls.constants(snapshot.header, init=False)
    cls.update_offsets(snapshot)

  def create(cls, snapshot, overwrite=True):
    if not overwrite:
      file = cls.file_class(snapshot.file, endian=cls.endian, mode='wx+')
    else:
      file = cls.file_class(snapshot.file, endian=cls.endian, mode='w+')
    snapshot.reader = cls
    snapshot.C = cls.create_header()
    snapshot.header = snapshot.C._header

  def create_header(cls):
    header = numpy.zeros(dtype=cls.header, shape=None)
    C = cls.constants(header, init=True)
    return C

  def __getitem__(cls, key):
    return cls.schema[key]
  def __contains__(cls, key):
    return key in cls.schema
  def __iter__(cls):
    for n in cls.schema:
      yield cls.schema[n]

  def update_offsets(cls, snapshot):
    blockpos = cls.file_class.get_size(cls.header.itemsize);
    for block in cls.schema:
      if not cls.has_block(snapshot, block):
        snapshot.sizes[block] = None
        snapshot.offsets[block] = None
        continue

      N = cls.count_particles(snapshot, block, None)
      blocksize = N * cls.schema[block].dtype.itemsize

      snapshot.sizes[block] = blocksize
      snapshot.offsets[block] = blockpos
      blockpos += cls.file_class.get_size(blocksize);
    return blockpos

  def has_block(cls, snapshot, block, ptype=None):
    if not block in cls.schema: return False
    schema = cls.schema[block]
    if schema.conditions is not None:
      for cond in schema.conditions:
        if snapshot.C[cond] == 0 : return False
    if ptype is None: return True

    if ptype in schema.ptypes: return True
    return False

  def count_particles(cls, snapshot, name, endptype=None):
    N = 0
    schema = cls.schema[name]
    if endptype is None:
      endptype = cls.schema.nptypes
    for ptype in range(endptype):
      if cls.needmasstab(snapshot, name, ptype):
        continue
      if ptype in schema.ptypes:
        N += int(snapshot.C['N'][ptype])
    return N

  def write_header(cls, snapshot):
    file = cls.file_class(snapshot.file, endian=cls.endian, mode='r+')
    file.seek(0)
    file.write_record(snapshot.header)
    
  def create_structure(cls, snapshot):
    file = cls.file_class(snapshot.file, endian=cls.endian, mode='r+')
    cls.update_offsets(snapshot)
    for s in cls:
        if not snapshot.sizes[s.name] == None:
          file.seek(snapshot.offsets[s.name])
          file.create_record(s.dtype.base, snapshot.sizes[s.name] // s.dtype.base.itemsize)
    file.seek(0)
    file.write_record(snapshot.header)

  def save(cls, snapshot, ptype, name):
    if not cls.has_block(snapshot, name, ptype):
      return

    if cls.needmasstab(snapshot, name, ptype):
      if (snapshot.P[ptype][name] != snapshot.C['mass'][ptype]).any():
        warnings.warn('mismatching particle mass detected')
      return

    file = cls.file_class(snapshot.file, endian=cls.endian, mode='r+')
    file.seek(snapshot.offsets[name])
    dtype = cls.schema[name].dtype
    length = snapshot.sizes[name] // dtype.base.itemsize
    offset = cls.count_particles(snapshot, name, ptype)
    offset *= dtype.itemsize // dtype.base.itemsize
    if len(snapshot.P[ptype][name]) != snapshot.C['N'][ptype]:
      raise IOError('snapshot memory image corrupted: %s %d %d %d' % (name, ptype, len(snapshot.P[ptype][name]), snapshot.C['N'][ptype]))
    file.write_record(snapshot.P[ptype][name], length, offset)
   
  def check(cls, snapshot):
    file = cls.file_class(snapshot.file, endian=cls.endian, mode='r')
    for s in cls:
      file.seek(snapshot.offsets[s.name])
      length = snapshot.sizes[s.name] // s.dtype.itemsize
      file.skip_record(s.dtype, length)

  def alloc(cls, snapshot, ptype, name):
    """ allocate memory for the block """
    dtype = cls.schema[name].dtype
    ptypes = cls.schema[name].ptypes
    if ptype is None:
      length = snapshot.sizes[name] // dtype.itemsize
      snapshot.P[None][name] = numpy.zeros(length, dtype)

      for ptype in ptypes:
        if cls.needmasstab(snapshot, name, ptype):
          snapshot.P[ptype][name] = numpy.empty(snapshot.C['N'][ptype], dtype)
          snapshot.P[ptype][name][:] = snapshot.C['mass'][ptype]
        else:
          offset = cls.count_particles(snapshot, name, ptype)
          snapshot.P[ptype][name] = snapshot.P[None][name]\
               [offset:offset + snapshot.C['N'][ptype]]
      return 

    if not cls.has_block(snapshot, name, ptype):
      snapshot.P[ptype][name] = None
      return

    if cls.needmasstab(snapshot, name, ptype):
      snapshot.P[ptype][name] = numpy.empty(snapshot.C['N'][ptype], dtype)
      snapshot.P[ptype][name][:] = snapshot.C['mass'][ptype]
    else:
      length = snapshot.C['N'][ptype]
      snapshot.P[ptype][name] = numpy.zeros(length, dtype)


  def load(cls, snapshot, ptype, name):
    """ if ptype is None, read in all ptypes """
    dtype = cls.schema[name].dtype
    ptypes = cls.schema[name].ptypes
    if ptype is None:
      file = cls.file_class(snapshot.file, endian=cls.endian, mode='r')
      file.seek(snapshot.offsets[name])
      length = snapshot.sizes[name] // dtype.itemsize
      try:
        snapshot.P[None][name] = file.read_record(dtype, length)
      except IOError as e:
        raise IOError('failed to read block %s:%s' % (s.name, e))

      for ptype in ptypes:
        if cls.needmasstab(snapshot, name, ptype):
          snapshot.P[ptype][name] = numpy.empty(snapshot.C['N'][ptype], dtype)
          snapshot.P[ptype][name][:] = snapshot.C['mass'][ptype]
        else:
          offset = cls.count_particles(snapshot, name, ptype)
          snapshot.P[ptype][name] = snapshot.P[None][name]\
               [offset:offset + snapshot.C['N'][ptype]]
      return

    if not cls.has_block(snapshot, name, ptype):
      snapshot.P[ptype][name] = None
      return

    if cls.needmasstab(snapshot, name, ptype):
      snapshot.P[ptype][name] = numpy.empty(snapshot.C['N'][ptype], dtype)
      snapshot.P[ptype][name][:] = snapshot.C['mass'][ptype]
    else:
      file = cls.file_class(snapshot.file, endian=cls.endian, mode='r')
      file.seek(snapshot.offsets[name])
      length = snapshot.sizes[name] // dtype.itemsize
      offset = cls.count_particles(snapshot, name, ptype)
      try:
        snapshot.P[ptype][name] = file.read_record(dtype, length, offset, snapshot.C['N'][ptype])
      except IOError as e:
        raise IOError('failed to read block %s:%s' % (name, e))
  def needmasstab(cls, snapshot, name, ptype):
    return cls.usemasstab and name == 'mass' \
          and snapshot.C['mass'][ptype] != 0.0
    
