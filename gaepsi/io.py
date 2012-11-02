import numpy
class CFile(file):
  @staticmethod
  def get_size(size):
    return size
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
  @staticmethod
  def get_size(size):
    if size == 0: return 0
    return size + 2 * 4

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

