from numpy import fromfile, dtype
from readers.io import F77File

header_dtype = dtype([
  ('ncell1d', 'u4'),
  ('ncells', 'u4'),
  ('cellsize', 'f4'),
  ('unused', ('u1', 256-12))])
class Meshmap:
  def __init__(self, file):
    if not hasattr(file, 'read_record'):
      file = F77File(file, 'r')
    header = file.read_record(dtype= header_dtype, length = 1)[0]
    self.ncell1d = header['ncell1d']
    self.cellsize = header['cellsize']
    self.ncells = header['ncells']
    self.cells = {}
    file.skip_record(dtype = dtype('u1'))
    self.Nfiles = file.read_record(dtype = dtype('u4'))
    mask = (self.Nfiles != 0)
    self.offsets = zeros(dtype='u8', shape = self.Nfiles.size)
    tmp = self.Nfiles.copy()
    tmp[mask] += 2
    self.offsets[1:] = tmp.cumsum()[0:self.ncells-1]
    self.offsets += 1
    self.buf = fromfile(file, dtype='i4')
  def cut2fid(self, cut):
    
  def get(self, icell):
    return self.buf[self.offsets[icell]:self.offsets[icell]+self.Nfiles[icell]]
    
