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

  def read_record(self, dtype, length = None, offset=None, nread=None) :
    if offset == None: offset = 0
    if nread == None: nread = length - offset
    self.seek(offset * dtype.itemsize, 1)
    arr = fromfile(self, dtype, length)
    self.seek((length - nread - offset) * dtype.itemsize, 1)
    return arr
  def skip_record(self, dtype, length) :
    size = length * dtype.itemsize
    self.seek(size, 1)
  def write_record(self, a):
    a.tofile(self)
  def rewind_record(self, dtype, length) :
    size = length * dtype.itemsize
    self.seek(-size, 1)

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

  def read_record(self, dtype, length = None, offset=None, nread=None) :
    if length == 0: return array([])
    size = fromfile(self, self.bsdtype, 1)[0]
    _length = size / dtype.itemsize;
    if length != None and length != _length:
      raise IOError("length doesn't match %d != %d" % (length, _length))
    
    length = _length
    if offset == None: offset = 0
    if nread == None: nread = length - offset
    self.seek(offset * dtype.itemsize, 1)
    X = fromfile(self, dtype, nread)
    self.seek((length - nread - offset) * dtype.itemsize, 1)
    size2 = fromfile(self, self.bsdtype, 1)[0]
    if size != size2 :
      raise IOError("record size doesn't match %d != %d" % (size, size2))
    if self.little_endian != little_endian: X.byteswap(True)
    return X

  def write_record(self, a):
    if a.size == 0: return
    size = int32(a.size * a.dtype.itemsize)
    array([size], dtype=self.bsdtype).tofile(self)
    
    if self.little_endian != little_endian: a.byteswap(True)
    a.tofile(self)
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
