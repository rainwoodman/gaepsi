import numpy
import warnings

def is_string_like(v):
  try: v + ''
  except: return False
  return True

def get_reader(reader):
  return Reader(reader)

def Reader(reader):
  if is_string_like(reader) :
    module = __import__('gaepsi.readers.%s' % reader, globals(), {}, [''], 0)
    if not hasattr(module, 'Reader'):
      raise ImportError('Reader class not found in %s', reader)
    reader = module.Reader
  reader = ReaderMeta(reader.__name__, (reader, object), dict(reader.__dict__))
  return reader

class Constants:
  """
    dict = {
      key: action
    }
    action can be:
      a basestring: index in header
      a list:       list of indices in header
      a length 1 tuple: a getter function
      a length 2 tuple: a getter function and a setter function
    getter function: lambda header: header[bluh] + header[bluh] ...
    setter function: lambda value: {key:value/100, key2:value+200} ...
  """
  def __init__(self, header, dict={}):
    self.header = header
    self.dict = dict

  def __iter__(self):
    return iter(self.dict)
  def __contains__(self, index):
    return index in self.dict

  def __repr__(self):
    return "Constants(header=%s, dict=%s)" % (repr(self.header), repr(self.dict))

  def __str__(self):
    items = ['%s = %s' % (index, self[index]) for index in self.dict]
    return '\n'.join(items)
    
  def __getitem__(self, index):
    if not index in self.dict:
      return self.header[index]
    action = self.dict[index]
    if isinstance(action, basestring):
      return self.header[action]
    elif isinstance(action, tuple):
      if len(action) >= 1 and hasattr(action[0], '__call__'):
        return action[0](self.header)
      else:
        raise Exception("do not understand tuple constant, use (getter, setter)")
    elif isinstance(action, list):
      return numpy.array([self.__getitem__(item) for item in action])
    else:
      return action

  def __setitem__(self, index, value):
    if not index in self.dict:
      self.header[index] = value
      return
    action = self.dict[index]
    if isinstance(action, basestring):
      self.header[action] = value
    elif isinstance(action, tuple):
      if len(action) == 2 and hasattr(action[1], '__call__'):
        d = action[1](value)
        for key in d:
          self.__setitem__(key, d[key])
      else:
        raise Exception("do not understand tuple constant, use (getter, setter)")
    elif isinstance(action, list):
      for item, v in zip(action, value):
        self.__setitem__(item, v)
    else:
      raise Exception("constant[%s] has action %s and is readonly", index, action)


class Schema:
  from collections import namedtuple
  Entry = namedtuple("Entry", ['name', 'dtype', 'ptypes', 'conditions'])
  def __init__(self, list):
    self.dict = {}
    self.list = []
    for entry in list:
      self.dict[entry[0]] = Schema.Entry._make((entry[0], numpy.dtype(entry[1]), entry[2], entry[3]))
      self.list += [entry[0]]
  def __contains__(self, index):
    return index in self.dict
  def __getitem__(self, index):
    return self.dict[index]
  def __iter__(self):
    return iter(self.list)

class ReaderMeta(type):
  def __new__(meta, name, base, dict):
    properties = {
      'format': 'F',
      'header': None,
      'schema': None,
      'defaults': {},
      'constants': {},
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
    cls.schema = Schema(cls.schema)


  def __str__(cls):
    return str(cls.__dict__)

  def open(cls, snapshot):
    file = cls.file_class(snapshot.file, endian=cls.endian, mode='r')
    snapshot.reader = cls
    file.seek(0)
    snapshot.header = file.read_record(cls.header, 1).squeeze()
    snapshot.C = Constants(snapshot.header, cls.constants)
    cls.update_offsets(snapshot)

  def create(cls, snapshot, overwrite=True):
    if not overwrite:
      file = cls.file_class(snapshot.file, endian=cls.endian, mode='wx+')
    else:
      file = cls.file_class(snapshot.file, endian=cls.endian, mode='w+')
    snapshot.reader = cls
    snapshot.header = numpy.zeros(dtype=cls.header, shape=None)
    for f in cls.defaults:
      snapshot.header[f] = cls.defaults[f]
    snapshot.C = Constants(snapshot.header, cls.constants)

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
    if ptype in s.ptypes: return True
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

      N = 0
      for ptype in s.ptypes:
        N += snapshot.C['N'][ptype]
      blocksize = N * s.dtype.itemsize

      snapshot.sizes[s.name] = blocksize
      snapshot.offsets[s.name] = blockpos
      if blocksize != 0: 
        blockpos += cls.file_class.get_size(blocksize);

    return blockpos

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
    file = cls.file_class(snapshot.file, endian=cls.endian, mode='r+')

    s = cls[name]
    if not ptype in s.ptypes: 
      return
    file.seek(snapshot.offsets[s.name])

    length = snapshot.sizes[s.name] // s.dtype.base.itemsize
    offset = 0
    for i in range(len(snapshot.C['N'])):
      if i in s.ptypes and i < ptype :
        offset += snapshot.C['N'][i]
    offset *= s.dtype.itemsize / s.dtype.base.itemsize
    file.write_record(snapshot.P[ptype][s.name], length, offset)
   
  def check(cls, snapshot):
    file = cls.file_class(snapshot.file, endian=cls.endian, mode='r')
    for s in cls:
      file.seek(snapshot.offsets[s.name])
      length = snapshot.sizes[s.name] // s.dtype.itemsize
      file.skip_record(s.dtype, length)

  def load(cls, snapshot, ptype, name):
    file = cls.file_class(snapshot.file, endian=cls.endian, mode='r')

    s = cls[name]
    file.seek(snapshot.offsets[s.name])
    length = snapshot.sizes[s.name] // s.dtype.itemsize
    if not ptype in s.ptypes : 
      snapshot.P[ptype][s.name] = None
      return
    offset = 0
    for i in range(len(snapshot.C['N'])):
      if i in s.ptypes and i < ptype :
        offset += snapshot.C['N'][i]
    snapshot.P[ptype][s.name] = file.read_record(s.dtype, length, offset, snapshot.C['N'][ptype])

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
      raise IOError("length doesn't match %d != %d" % (length, _length))
    
    length = _length
    if nread == None: nread = length - offset
    self.seek(offset * dtype.itemsize, 1)
    X = numpy.fromfile(self, dtype, nread)
    self.seek((length - nread - offset) * dtype.itemsize, 1)
    size2 = numpy.fromfile(self, self.bsdtype, 1)[0]
    if size != size2 :
      raise IOError("record size doesn't match %d != %d" % (size, size2))
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
      raise IOError("length doesn't match %d != %d" % (length, _length))
    self.seek(size, 1)
    size2 = numpy.fromfile(self, self.bsdtype, 1)[0]
    if size != size2 :
      raise IOError("record size doesn't match %d != %d" % (size, size2))

  def rewind_record(self, dtype, length = None) :
    dtype = numpy.dtype(dtype)
    if length == 0: return
    self.seek(-self.bsdtype.itemsize, 1)
    size = numpy.fromfile(self, self.bsdtype, 1)[0]
    _length = size / dtype.itemsize;
    if length != None and length != _length:
      raise IOError("length doesn't match %d != %d" % (length, _length))
    self.seek(-size, 1)
    self.seek(-self.bsdtype.itemsize, 1)
    size2 = numpy.fromfile(self, self.bsdtype, 1)[0]
    self.seek(-self.bsdtype.itemsize, 1)
    if size != size2 :
      raise IOError("record size doesn't match %d != %d" % (size, size2))
  def create_record(self, dtype, length):
    dtype = numpy.dtype(dtype)
    size = numpy.int32(length * dtype.itemsize)
    numpy.array([size], dtype=self.bsdtype).tofile(self)
    self.seek(size, 1)
    numpy.array([size], dtype=self.bsdtype).tofile(self)
