import numpy

class CFile(file):
  def __init__(self, *args, **kwargs) :
    file.__init__(self, *args, **kwargs)
  def read_record(self, dtype, length) :
    return numpy.fromfile(self, dtype, length)
  def skip_record(self, dtype, length) :
    size = length * dtype.itemsize
    self.seek(size, 1)
  def rewind_record(self, dtype, length) :
    size = length * dtype.itemsize
    self.seek(-size, 1)
  
class F77File(file):
  def __init__(self, *args, **kwargs) :
    file.__init__(self, *args, **kwargs)

  def read_record(self, dtype, length = None) :
    size = numpy.fromfile(self, 'i4', 1)[0]
    _length = size / dtype.itemsize;
    if length != None and length != _length:
      raise IOError("length doesn't match %d != %d" % (length, _length))
    
    length = _length
    X = numpy.fromfile(self, dtype, length)
    size2 = numpy.fromfile(self, 'i4', 1)[0]
    if size != size2 :
      raise IOError("record size doesn't match %d != %d" % (size, size2))
    return X

  def skip_record(self, dtype, length = None) :
    size = numpy.fromfile(self, 'i4', 1)[0]
    _length = size / dtype.itemsize;
    if length != None and length != _length:
      raise IOError("length doesn't match %d != %d" % (length, _length))
    self.seek(size, 1)
    size2 = numpy.fromfile(self, 'i4', 1)[0]
    if size != size2 :
      raise IOError("record size doesn't match %d != %d" % (size, size2))

  def rewind_record(self, dtype, length = None) :
    self.seek(-dtype('i4').itemsize, 1)
    size = numpy.fromfile(self, 'i4', 1)[0]
    _length = size / dtype.itemsize;
    if length != None and length != _length:
      raise IOError("length doesn't match %d != %d" % (length, _length))
    self.seek(-size, 1)
    self.seek(-dtype('i4').itemsize, 1)
    size2 = numpy.fromfile(self, 'i4', 1)[0]
    self.seek(-dtype('i4').itemsize, 1)
    if size != size2 :
      raise IOError("record size doesn't match %d != %d" % (size, size2))
