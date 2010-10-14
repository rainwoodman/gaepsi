from numpy import fromfile
from numpy import int32
from numpy import array
from numpy import dtype

class CFile(file):
  def __init__(self, *args, **kwargs) :
    file.__init__(self, *args, **kwargs)
  def read_record(self, dtype, length) :
    return fromfile(self, dtype, length)
  def skip_record(self, dtype, length) :
    size = length * dtype.itemsize
    self.seek(size, 1)
  def write_record(self, a):
    a.tofile(self)
  def rewind_record(self, dtype, length) :
    size = length * dtype.itemsize
    self.seek(-size, 1)
  
class F77File(file):
  def __init__(self, *args, **kwargs) :
    file.__init__(self, *args, **kwargs)

  def read_record(self, dtype, length = None) :
    size = fromfile(self, 'i4', 1)[0]
    _length = size / dtype.itemsize;
    if length != None and length != _length:
      raise IOError("length doesn't match %d != %d" % (length, _length))
    
    length = _length
    X = fromfile(self, dtype, length)
    size2 = fromfile(self, 'i4', 1)[0]
    if size != size2 :
      raise IOError("record size doesn't match %d != %d" % (size, size2))
    return X

  def write_record(self, a):
    size = int32(a.size * a.dtype.itemsize)
    array([size], dtype='i4').tofile(self)
    a.tofile(self)
    array([size], dtype='i4').tofile(self)

  def skip_record(self, dtype, length = None) :
    size = fromfile(self, 'i4', 1)[0]
    _length = size / dtype.itemsize;
    if length != None and length != _length:
      raise IOError("length doesn't match %d != %d" % (length, _length))
    self.seek(size, 1)
    size2 = fromfile(self, 'i4', 1)[0]
    if size != size2 :
      raise IOError("record size doesn't match %d != %d" % (size, size2))

  def rewind_record(self, dtype, length = None) :
    self.seek(-dtype('i4').itemsize, 1)
    size = fromfile(self, 'i4', 1)[0]
    _length = size / dtype.itemsize;
    if length != None and length != _length:
      raise IOError("length doesn't match %d != %d" % (length, _length))
    self.seek(-size, 1)
    self.seek(-dtype('i4').itemsize, 1)
    size2 = fromfile(self, 'i4', 1)[0]
    self.seek(-dtype('i4').itemsize, 1)
    if size != size2 :
      raise IOError("record size doesn't match %d != %d" % (size, size2))
