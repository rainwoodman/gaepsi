import numpy
import warnings
from gaepsi.tools import virtarray
from types import MethodType
def is_string_like(v):
  try: v + ''
  except: return False
  return True

def get_reader(reader):
  return Reader(reader)

def Reader(reader, forcedouble=False):
  if is_string_like(reader) :
    module = __import__('gaepsi.readers.%s' % reader, globals(), {}, [''], 0)
    if not hasattr(module, 'Reader'):
      raise ImportError('Reader class not found in %s', reader)
    reader = module.Reader
  if not isinstance(reader, ReaderMeta):
    reader = ReaderMeta(reader.__name__, (reader, object), dict(reader.__dict__))
  if forcedouble:
    reader.schema.force_double_precision()

  return reader

class ConstBase:
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
    def func():
      for name in sorted(numpy.unique(dir(self) \
         + list(self._header.dtype.names) \
         + list(self._extra.keys()))):
        if name[0] == '_': continue
        yield name
    return func()
    
  def __contains__(self, item):
    return item in self._header.dtype.names or hasattr(self, item) or item in self._extra

  def __getitem__(self, item):
    if hasattr(self, item):
      attr = getattr(self, item)
      if isinstance(attr, basestring):
        return self._header[attr]
      elif isinstance(attr, tuple) or isinstance(attr, list):
        dtype, get, set = attr
        return virtarray(None, dtype, MethodType(get, self, None), MethodType(set, self, None))
      else:
        return attr
    elif item in self._header.dtype.names:
      return self._header[item]
    else:
      return self._extra[item]

  def __setitem__(self, item, value):
    if hasattr(self, item):
      attr = getattr(self, item)
      if isinstance(attr, basestring):
        self._header[attr] = value
      elif isinstance(attr, tuple) or isinstance(attr, list):
        dtype, get, set = attr
        virtarray(None, dtype, MethodType(get, self, None), MethodType(set, self, None))[...] = value
      else:
        raise IndexError("can't set %s" % item)
    elif item in self._header.dtype.names:
      self._header[item] = value
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
  def __init__(self, list):
    self.dict = {}
    self.list = []
    for entry in list:
      if len(entry) == 3:
        conditions = []
      else:
        conditions = entry[3]
      self.dict[entry[0]] = Schema.Entry._make((entry[0], numpy.dtype(entry[1]), entry[2], conditions))
      self.list += [entry[0]]

  def force_single_precision(self):
    for name in self.list:
      entry = self.dict[name]
      self.dict[name] = Schema.Entry._make((
         name, eval(str(entry[1]).replace('f8', 'f4').replace('float64', 'float32'), {'numpy':numpy}),
         entry[2], entry[3]))
     
  def force_double_precision(self):
    for name in self.list:
      entry = self.dict[name]
      self.dict[name] = Schema.Entry._make((
         name, eval(repr(entry[1]).replace('f4', 'f8').replace('float32', 'float64'), {'dtype':numpy.dtype}),
         entry[2], entry[3]))
    
  def __contains__(self, index):
    return index in self.dict
  def __getitem__(self, index):
    return self.dict[index]
  def __iter__(self):
    return iter(self.list)
  def __str__(self):
    return str(self.list)

class ReaderMeta(type):
  def __new__(meta, name, base, dict):
    properties = {
      'format': 'F',
      'header': None,
      'schema': None,
      'constants': {},
      'usemasstab': True,
      'endian': '<'}

    missing = []
    dirs = []
    for x in base:
      dirs += dir(x)

    for p in properties:
      if not p in dict and not p in dirs:
        if properties[p] is not None:
          dict[p] = properties[p]
        else:
          missing += [p]

    if missing:
      raise ValueError("missing class properties in %s: %s" %
       (name, ', '.join(missing)))

    def _do_not_instantiate(cls):
      raise TypeError('%s is not instantiatable' % repr(cls))
    dict['__init__'] = _do_not_instantiate
    return type.__new__(meta, name, base, dict)

  def __init__(cls, name, base, dict):
    filedict = {'F': F77File, 'C': CFile }
    cls.file_class = filedict[cls.format]
    cls.header = numpy.dtype(cls.header)
    if not isinstance(cls.schema, Schema):
      cls.schema = Schema(cls.schema)
    if not issubclass(cls.constants, ConstBase):
      # not sure if the 'if' is useful. in no case it shall be already
      # a sublcass of ConstBase.
      cls.constants = type('constants', (cls.constants, ConstBase, object), {})
    else:
      raise

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
    snapshot.header = numpy.zeros(dtype=cls.header, shape=None)
    snapshot.C = cls.constants(snapshot.header, init=True)

  def __getitem__(cls, key):
    return cls.schema[key]
  def __contains__(cls, key):
    return key in cls.schema
  def __iter__(cls):
    for n in cls.schema:
      yield cls.schema[n]

  def has_block(cls, snapshot, ptype, block):
    if not block in cls: return False
    s = cls[block]
    for cond in s.conditions:
      if snapshot.header[cond] == 0 : return False
    if ptype is None or ptype in s.ptypes: return True
    return False

  def update_offsets(cls, snapshot):
    blockpos = cls.file_class.get_size(cls.header.itemsize);
    for s in cls:
      skip = False
      for cond in s.conditions:
        if snapshot.header[cond] == 0 : skip = True

      if skip :
        snapshot.sizes[s.name] = None
        snapshot.offsets[s.name] = None
        continue

      N = cls.count_particles(snapshot, s, None)
      blocksize = N * s.dtype.itemsize

      snapshot.sizes[s.name] = blocksize
      snapshot.offsets[s.name] = blockpos
      if blocksize != 0: 
        blockpos += cls.file_class.get_size(blocksize);

    return blockpos

  def count_particles(cls, snapshot, schema, endptype=None):
    N = 0
    for i in range(len(snapshot.C['N'])):
      if cls.usemasstab and schema.name == 'mass' and snapshot.C['mass'][i] != 0:
        continue
      if i in schema.ptypes and (endptype is None or i < endptype) :
        N += snapshot.C['N'][i]
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
    s = cls[name]
    if not ptype in s.ptypes: 
      return
    if cls.usemasstab and s.name == 'mass' and snapshot.C['mass'][ptype] != 0.0:
      if (snapshot.P[ptype][s.name] != snapshot.C['mass'][ptype]).any():
        warnings.warn('mismatching particle mass detected')
      return

    file = cls.file_class(snapshot.file, endian=cls.endian, mode='r+')
    file.seek(snapshot.offsets[s.name])
    length = snapshot.sizes[s.name] // s.dtype.base.itemsize
    offset = cls.count_particles(snapshot, s, ptype)
    offset *= s.dtype.itemsize / s.dtype.base.itemsize
    file.write_record(snapshot.P[ptype][s.name], length, offset)
   
  def check(cls, snapshot):
    file = cls.file_class(snapshot.file, endian=cls.endian, mode='r')
    for s in cls:
      file.seek(snapshot.offsets[s.name])
      length = snapshot.sizes[s.name] // s.dtype.itemsize
      file.skip_record(s.dtype, length)

  def load(cls, snapshot, ptype, name):
    """ if ptype is None, read in all ptypes """
    s = cls[name]
    if ptype is None:
      file = cls.file_class(snapshot.file, endian=cls.endian, mode='r')
      file.seek(snapshot.offsets[s.name])
      length = snapshot.sizes[s.name] // s.dtype.itemsize
      try:
        snapshot.P[None][s.name] = file.read_record(s.dtype, length)
      except IOError as e:
        raise IOError('failed to read block %s:%s' % (s.name, e))
      for ptype in s.ptypes:
        if cls.usemasstab and s.name == 'mass' and snapshot.C['mass'][ptype] != 0.0:
          snapshot.P[ptype][s.name] = numpy.empty(snapshot.C['N'][ptype], s.dtype)
          snapshot.P[ptype][s.name][:] = snapshot.C['mass'][ptype]
        else:
          offset = cls.count_particles(snapshot, s, ptype)
          snapshot.P[ptype][s.name] = snapshot.P[None][s.name]\
               [offset:offset + snapshot.C['N'][ptype]]

    if not ptype in s.ptypes: 
      snapshot.P[ptype][s.name] = None
      return
    if cls.usemasstab and s.name == 'mass' and snapshot.C['mass'][ptype] != 0.0:
      snapshot.P[ptype][s.name] = numpy.empty(snapshot.C['N'][ptype], s.dtype)
      snapshot.P[ptype][s.name][:] = snapshot.C['mass'][ptype]
    else:
      file = cls.file_class(snapshot.file, endian=cls.endian, mode='r')
      file.seek(snapshot.offsets[s.name])
      length = snapshot.sizes[s.name] // s.dtype.itemsize
      offset = cls.count_particles(snapshot, s, ptype)
      try:
        snapshot.P[ptype][s.name] = file.read_record(s.dtype, length, offset, snapshot.C['N'][ptype])
      except IOError as e:
        raise IOError('failed to read block %s:%s' % (s.name, e))

class CFile(file):
  def get_size(size):
    return size
  get_size = staticmethod(get_size)
  def __init__(self, *args, **kwargs) :
    self.endian = kwargs.pop('endian', 'N')
    self.bsdtype = numpy.dtype('i4').newbyteorder(self.endian)
    self.little_endian = ( self.bsdtype.byteorder == '<' or (
                     self.bsdtype.byteorder == '=' and numpy.little_endian))
    file.__init__(self, *args, **kwargs)

  def read_record(self, dtype, length = None, offset=0, nread=None) :
    dtype = numpy.dtype(dtype)
    if nread == None: nread = length - offset
    self.seek(offset * dtype.itemsize, 1)
    arr = numpy.fromfile(self, dtype, length)
    self.seek((length - nread - offset) * dtype.itemsize, 1)
    return arr
  def skip_record(self, dtype, length) :
    dtype = numpy.dtype(dtype)
    size = length * dtype.itemsize
    self.seek(size, 1)
  def write_record(self, a, length = None, offset=0):
    dtype = a.dtype
    self.seek(offset * dtype.itemsize, 1)
    a.tofile(self)
    self.seek((length - a.size - offset) * dtype.itemsize, 1)
  def rewind_record(self, dtype, length) :
    dtype = numpy.dtype(dtype)
    size = length * dtype.itemsize
    self.seek(-size, 1)
  def create_record(self, dtype, length):
    dtype = numpy.dtype(dtype)
    self.seek(length * dtype.itemsize, 1)

class F77File(file):
  def get_size(size):
    if size == 0: return 0
    return size + 2 * 4
  get_size = staticmethod(get_size)

  def __init__(self, *args, **kwargs) :
    self.endian = kwargs.pop('endian', 'N')
    
    self.bsdtype = numpy.dtype('i4').newbyteorder(self.endian)
    self.little_endian = ( self.bsdtype.byteorder == '<' or (
                     self.bsdtype.byteorder == '=' and numpy.little_endian))
    file.__init__(self, *args, **kwargs)

  def read_record(self, dtype, length = None, offset=0, nread=None) :
    dtype = numpy.dtype(dtype)
    if length == 0: return numpy.array([], dtype=dtype)
    size = numpy.fromfile(self, self.bsdtype, 1)[0]
    _length = size / dtype.itemsize;
    if length != None and length != _length:
      raise IOError("length doesn't match %d(expect) != %d(real)" % (length, _length))
    
    length = _length
    if nread == None: nread = length - offset
    self.seek(offset * dtype.itemsize, 1)
    X = numpy.fromfile(self, dtype, nread)
    self.seek((length - nread - offset) * dtype.itemsize, 1)
    size2 = numpy.fromfile(self, self.bsdtype, 1)[0]
    if size != size2 :
      raise IOError("record size doesn't match %d(expect) != %d(real)" % (size, size2))
    if self.little_endian != numpy.little_endian: X.byteswap(True)
    return X

  def write_record(self, a, length = None, offset=0):
    if length == None: length = a.size

    if length == 0: return
    if self.little_endian != numpy.little_endian: a.byteswap(True)
    dtype = a.dtype
    size = numpy.int32(length * a.dtype.itemsize)
    numpy.array([size], dtype=self.bsdtype).tofile(self)
    self.seek(offset * dtype.itemsize, 1)
    a.tofile(self)
    self.seek((length - offset - a.size) * dtype.itemsize, 1)
    numpy.array([size], dtype=self.bsdtype).tofile(self)

  def skip_record(self, dtype, length = None) :
    dtype = numpy.dtype(dtype)
    if length == 0: return
    size = numpy.fromfile(self, self.bsdtype, 1)[0]
    _length = size / dtype.itemsize;
    if length != None and length != _length:
      raise IOError("length doesn't match %d(expect) != %d(real)" % (length, _length))
    self.seek(size, 1)
    size2 = numpy.fromfile(self, self.bsdtype, 1)[0]
    if size != size2 :
      raise IOError("record size doesn't match %d(expect) != %d(real)" % (size, size2))

  def rewind_record(self, dtype, length = None) :
    dtype = numpy.dtype(dtype)
    if length == 0: return
    self.seek(-self.bsdtype.itemsize, 1)
    size = numpy.fromfile(self, self.bsdtype, 1)[0]
    _length = size / dtype.itemsize;
    if length != None and length != _length:
      raise IOError("length doesn't match %d(expect) != %d(real)" % (length, _length))
    self.seek(-size, 1)
    self.seek(-self.bsdtype.itemsize, 1)
    size2 = numpy.fromfile(self, self.bsdtype, 1)[0]
    self.seek(-self.bsdtype.itemsize, 1)
    if size != size2 :
      raise IOError("record size doesn't match %d(expect) != %d(real)" % (size, size2))

  def create_record(self, dtype, length):
    dtype = numpy.dtype(dtype)
    size = numpy.int32(length * dtype.itemsize)
    numpy.array([size], dtype=self.bsdtype).tofile(self)
    self.seek(size, 1)
    numpy.array([size], dtype=self.bsdtype).tofile(self)
