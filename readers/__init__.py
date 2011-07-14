from numpy import dtype
from numpy import zeros
class Constants:
  def __init__(self, reader, header):
    self.header = header
    self.reader = reader
  def __getattr__(self, index):
    return self.reader.get_constant(self.header, index)
  def __getitem__(self, index):
    return self.reader.get_constant(self.header, index)
  def __setitem__(self, index, value):
    self.reader.set_constant(self.header, index, value)

class ReaderBase:
  def __init__(self, file_class, header, schemas, constants={}, endian='<'):
    self.header_dtype = dtype(header)
    self.constants = constants
    self.schemas = [dict(name = sch[0], 
                    dtype=dtype(sch[1]),
                    ptypes=sch[2], 
                    conditions=sch[3]) for sch in schemas]
    self.file_class = file_class
    self.endian = endian
    self.hash = {}
    for s in self.schemas:
      self.hash[s['name']] = s

  def open(self, snapshot, file, *args, **kwargs):
    snapshot.file = self.file_class(file, endian=self.endian, *args, **kwargs)
    snapshot.reader = self
    self.load(snapshot, name = 'header')
    snapshot.C = Constants(self, snapshot.header)
    self.update_offsets(snapshot)

  def create(self, snapshot, file):
    snapshot.file = self.file_class(file, endian=self.endian, mode='w+')
    snapshot.reader = self
    buf = zeros(dtype=self.header_dtype, shape=1)
    snapshot.header = buf[0]
    snapshot.C = Constants(self, snapshot.header)

  def get_constant(self, header, index):
    entry = self.constants[index]
    if hasattr(entry, 'isalnum'):
      return header[entry]
    return entry

  def set_constant(self, header, index, value):
    entry = self.constants[index]
    if hasattr(entry, 'isalnum'):
      header[entry] = value
      return
    raise IndexError('%s is readonly', index)

  def __getitem__(self, key):
    return self.hash[key]
  def __contains__(self, key):
    return key in self.hash
  def constants(self, snapshot):
    return dict(
      N = snapshot.header['N'])

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
        N += snapshot.C.N[ptype]
      blocksize = N * s['dtype'].itemsize

      snapshot.sizes[name] = blocksize
      snapshot.offsets[name] = blockpos
      if blocksize != 0 : 
        blockpos += self.file_class.get_size(blocksize);

    return blockpos


  def save(self, snapshot, name, ptype='all'):
    if name == 'header':
      self.update_offsets(snapshot)
      for s in self.schemas:
        name = s['name']
        if not snapshot.sizes[name] == None:
          snapshot.file.seek(snapshot.offsets[name])
          snapshot.file.create_record(s['dtype'], snapshot.sizes[name] // s['dtype'].itemsize)
      snapshot.file.seek(0)
      buf = zeros(dtype = self.header_dtype, shape = 1)
      buf[0] = snapshot.header
      snapshot.file.write_record(buf, 1)
      snapshot.file.flush()
      return

    sch = self.hash[name]
    snapshot.file.seek(snapshot.offsets[name])
    length = snapshot.sizes[name] // sch['dtype'].itemsize
    if ptype == 'all':
      if snapshot.sizes[name] != 0 :
        snapshot.file.write_record(snapshot.P['all'][name])
        snapshot.file.flush()
    else :
      if not ptype in sch['ptypes'] : 
        return
      offset = 0
      for i in range(6):
        if i in sch['ptypes'] and i < ptype :
          offset += snapshot.C.N[i]
      snapshot.file.write_record(snapshot.P[ptype][name], length, offset)
      snapshot.file.flush()
   
  def check(self, snapshot):
    for sch in self.schemas:
      name = sch['name']
      snapshot.file.seek(snapshot.offsets[name])
      length = snapshot.sizes[name] // sch['dtype'].itemsize
      snapshot.file.skip_record(sch['dtype'], length)
   

  def load(self, snapshot, name, ptype='all'):
    if name == 'header':
      snapshot.file.seek(0)
      snapshot.header = snapshot.file.read_record(self.header_dtype, 1)[0]
      return

    if snapshot[ptype].has_key(name) : return

    sch = self.hash[name]
    snapshot.file.seek(snapshot.offsets[name])
    length = snapshot.sizes[name] // sch['dtype'].itemsize
    if ptype == 'all':
      if snapshot.sizes[name] != 0 :
        snapshot.P['all'][name] = snapshot.file.read_record(sch['dtype'], length)
      else :
        snapshot.P['all'][name] = None
    else :
      if not ptype in sch['ptypes'] : 
        snapshot.P[ptype][name] = None
        return
      offset = 0
      for i in range(6):
        if i in sch['ptypes'] and i < ptype :
          offset += snapshot.C.N[i]
      snapshot.P[ptype][name] = snapshot.file.read_record(sch['dtype'], length, offset, snapshot.C.N[ptype])

from numpy import fromfile
from numpy import int32
from numpy import array
from numpy import dtype
from numpy import little_endian
class CFile(file):
  def get_size(size):
    return size
  get_size = staticmethod(get_size)
  def __init__(self, *args, **kwargs) :
    file.__init__(self, *args, **kwargs)

  def read_record(self, dtype, length = None, offset=0, nread=None) :
    if nread == None: nread = length - offset
    self.seek(offset * dtype.itemsize, 1)
    arr = fromfile(self, dtype, length)
    self.seek((length - nread - offset) * dtype.itemsize, 1)
    return arr
  def skip_record(self, dtype, length) :
    size = length * dtype.itemsize
    self.seek(size, 1)
  def write_record(self, a, length = None, offset=0):
    dtype = a.dtype
    self.seek(offset * dtype.itemsize, 1)
    a.tofile(self)
    self.seek((length - a.size - offset) * dtype.itemsize, 1)
  def rewind_record(self, dtype, length) :
    size = length * dtype.itemsize
    self.seek(-size, 1)
  def create_record(self, dtype, length):
    self.seek(length * dtype.itemsize, 1)

class F77File(file):
  def get_size(size):
    if size == 0: return 0
    return size + 2 * 4
  get_size = staticmethod(get_size)

  def __init__(self, *args, **kwargs) :
    try:
      self.endian = kwargs['endian']
      del kwargs['endian']
    except KeyError:
      self.endian = 'N'
    
    self.bsdtype = dtype('i4').newbyteorder(self.endian)
    self.little_endian = ( self.bsdtype.byteorder == '<' or (
                     self.bsdtype.byteorder == '=' and little_endian))
    file.__init__(self, *args, **kwargs)

  def read_record(self, dtype, length = None, offset=0, nread=None) :
    if length == 0: return array([])
    size = fromfile(self, self.bsdtype, 1)[0]
    _length = size / dtype.itemsize;
    if length != None and length != _length:
      raise IOError("length doesn't match %d != %d" % (length, _length))
    
    length = _length
    if nread == None: nread = length - offset
    self.seek(offset * dtype.itemsize, 1)
    X = fromfile(self, dtype, nread)
    self.seek((length - nread - offset) * dtype.itemsize, 1)
    size2 = fromfile(self, self.bsdtype, 1)[0]
    if size != size2 :
      raise IOError("record size doesn't match %d != %d" % (size, size2))
    if self.little_endian != little_endian: X.byteswap(True)
    return X

  def write_record(self, a, length = None, offset=0):
    if length == None: length = a.size

    if length == 0: return
    if self.little_endian != little_endian: a.byteswap(True)
    dtype = a.dtype
    size = int32(length * a.dtype.itemsize)
    array([size], dtype=self.bsdtype).tofile(self)
    self.seek(offset * dtype.itemsize, 1)
    a.tofile(self)
    self.seek((length - offset - a.size) * dtype.itemsize, 1)
    array([size], dtype=self.bsdtype).tofile(self)

  def skip_record(self, dtype, length = None) :
    if length == 0: return
    size = fromfile(self, self.bsdtype, 1)[0]
    _length = size / dtype.itemsize;
    if length != None and length != _length:
      raise IOError("length doesn't match %d != %d" % (length, _length))
    self.seek(size, 1)
    size2 = fromfile(self, self.bsdtype, 1)[0]
    if size != size2 :
      raise IOError("record size doesn't match %d != %d" % (size, size2))

  def rewind_record(self, dtype, length = None) :
    if length == 0: return
    self.seek(-self.bsdtype.itemsize, 1)
    size = fromfile(self, self.bsdtype, 1)[0]
    _length = size / dtype.itemsize;
    if length != None and length != _length:
      raise IOError("length doesn't match %d != %d" % (length, _length))
    self.seek(-size, 1)
    self.seek(-self.bsdtype.itemsize, 1)
    size2 = fromfile(self, self.bsdtype, 1)[0]
    self.seek(-self.bsdtype.itemsize, 1)
    if size != size2 :
      raise IOError("record size doesn't match %d != %d" % (size, size2))
  def create_record(self, dtype, length):
    size = int32(length * dtype.itemsize)
    array([size], dtype=self.bsdtype).tofile(self)
    self.seek(size, 1)
    array([size], dtype=self.bsdtype).tofile(self)
