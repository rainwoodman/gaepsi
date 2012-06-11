import numpy
import warnings

def is_string_like(v):
  try: v + ''
  except: return False
  return True

def get_reader(reader):
  if is_string_like(reader) :
    try:
      _temp = __import__('gaepsi.readers.%s' % reader, globals(), locals(),
            ['Reader'],  -1)
    except ImportError:
      _temp = __import__(reader, globals(), locals(), ['Reader'], -1) 

    reader = _temp.Reader()
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
    return ind in self.dict

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

#  def __getattr__(self, index):
#    return self.__getitem__(index)
#  def __setattr__(self, index, value):
#    self.__setitem__(index, value)

class ReaderBase:
  def __init__(self, file_class, header, schemas, defaults={}, constants={}, endian='<'):
    """file_class is either F77File or CFile,
       header is a numpy dtype describing the header block
       schemas is a list of blocks, (name, dtype, [ptypes], [conditions]),
          for example ('pos', ('f4', 3), [0, 1, 4, 5], []), 
          or ('met', 'f4', [4], ['flags_hasmet'])
       defaults is a dictionary containing the default values used in the header, when a new snapshot file is created.
       constants is a dictionary of constant values/ corresponding header fields.
          for example, {'N': 'N', 'OmegaB': 0.044}
       endian is either '<'(intel) or '>' (ibm).
    """

    self.header_dtype = numpy.dtype(header)
    self.constants = constants
    self.schemas = [dict(name = sch[0], 
                    dtype=numpy.dtype(sch[1]),
                    ptypes=sch[2], 
                    conditions=sch[3]) for sch in schemas]
    self.file_class = file_class
    self.endian = endian
    self.hash = {}
    self.defaults = defaults
    for s in self.schemas:
      self.hash[s['name']] = s

  def open(self, snapshot):
    file = self.file_class(snapshot.file, endian=self.endian, mode='r')
    snapshot.reader = self

    file.seek(0)
    snapshot.header = file.read_record(self.header_dtype, 1).squeeze()

    snapshot.C = Constants(snapshot.header, self.constants)
    self.update_offsets(snapshot)

  def create(self, snapshot, overwrite=True):
    if not overwrite:
      file = self.file_class(snapshot.file, endian=self.endian, mode='wx+')
    else:
      file = self.file_class(snapshot.file, endian=self.endian, mode='w+')
    snapshot.reader = self
    snapshot.header = numpy.zeros(dtype=self.header_dtype, shape=None)
    for f in self.defaults:
      snapshot.header[f] = self.defaults[f]

    snapshot.C = Constants(snapshot.header, self.constants)

  def __getitem__(self, key):
    return self.hash[key]
  def __contains__(self, key):
    return key in self.hash

  def has_block(self, snapshot, ptype, block):
    if not block in self.hash: return False
    s = self.hash[block]
    for cond in s['conditions']:
      if snapshot.header[cond] == 0 : return False
    if ptype in s['ptypes']: return True
    return False

  def update_offsets(self, snapshot):
    blockpos = self.file_class.get_size(self.header_dtype.itemsize);
    for s in self.schemas:
      name = s['name']
      cease_existing = False
      for cond in s['conditions']:
        if snapshot.header[cond] == 0 : cease_existing = True

      if cease_existing :
        snapshot.sizes[name] = None
        snapshot.offsets[name] = None
        continue
      N = 0
      for ptype in s['ptypes']:
        N += snapshot.C['N'][ptype]
      blocksize = N * s['dtype'].itemsize

      snapshot.sizes[name] = blocksize
      snapshot.offsets[name] = blockpos
      if blocksize != 0 : 
        blockpos += self.file_class.get_size(blocksize);

    return blockpos


  def create_structure(self, snapshot):
    file = self.file_class(snapshot.file, endian=self.endian, mode='r+')
    self.update_offsets(snapshot)
    for s in self.schemas:
        name = s['name']
        if not snapshot.sizes[name] == None:
          file.seek(snapshot.offsets[name])
    # NOTE: for writing, because write_record sees only the base type of the dtype, we use the length from the basetype
          file.create_record(s['dtype'].base, snapshot.sizes[name] // s['dtype'].base.itemsize)
    file.seek(0)
    file.write_record(snapshot.header)

  def save(self, snapshot, ptype, name):
    file = self.file_class(snapshot.file, endian=self.endian, mode='r+')

    sch = self.hash[name]
    file.seek(snapshot.offsets[name])
    # NOTE: for writing, because write_record sees only the base type of the dtype, we use the length from the basetype
    length = snapshot.sizes[name] // sch['dtype'].base.itemsize
    if ptype == 'all':
      if snapshot.sizes[name] != 0 :
        file.write_record(snapshot.P['all'][name])
    else :
      if not ptype in sch['ptypes'] : 
        return
      offset = 0
      for i in range(len(snapshot.C['N'])):
        if i in sch['ptypes'] and i < ptype :
          offset += snapshot.C['N'][i]
      offset *= sch['dtype'].itemsize / sch['dtype'].base.itemsize
      file.write_record(snapshot.P[ptype][name], length, offset)
   
  def check(self, snapshot):
    file = self.file_class(snapshot.file, endian=self.endian, mode='r')
    for sch in self.schemas:
      name = sch['name']
      file.seek(snapshot.offsets[name])
      length = snapshot.sizes[name] // sch['dtype'].itemsize
      file.skip_record(sch['dtype'], length)
   

  def load(self, snapshot, ptype, name):
    file = self.file_class(snapshot.file, endian=self.endian, mode='r')

    sch = self.hash[name]
    file.seek(snapshot.offsets[name])
    length = snapshot.sizes[name] // sch['dtype'].itemsize
    if ptype == 'all':
      if snapshot.sizes[name] != 0 :
        snapshot.P['all'][name] = file.read_record(sch['dtype'], length)
      else :
        snapshot.P['all'][name] = None
    else :
      if not ptype in sch['ptypes'] : 
        snapshot.P[ptype][name] = None
        return
      offset = 0
      for i in range(len(snapshot.C['N'])):
        if i in sch['ptypes'] and i < ptype :
          offset += snapshot.C['N'][i]
      snapshot.P[ptype][name] = file.read_record(sch['dtype'], length, offset, snapshot.C['N'][ptype])

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
