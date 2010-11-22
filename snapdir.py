import io
from snapshot import Snapshot
from numpy import dtype
from numpy import floor, ceil
from numpy import int32

class Snapmap:
  def __init__(self, file):
    self.file = io.F77File(file)
    dt = dtype([
      ('Ncell1d', 'u4'),
      ('Ncells', 'u4'),
      ('cellsize', 'f4'),
      ('unused0', ('u4', 3)),
      ('mass', ('f8', 6)),
      ('time', 'f8'),
      ('redshift', 'f8'),
      ('flag_sfr', 'i4'),
      ('flag_feedback', 'i4'),
      ('Nparticle_total_low', ('u4', 6)),
      ('flag_cool', 'i4'),
      ('Nfiles', 'i4'),
      ('boxsize', 'f8'),
      ('OmegaM', 'f8'),
      ('OmegaL', 'f8'),
      ('h', 'f8'),
      ('flag_sft', 'i4'),
      ('flag_met', 'i4'),
      ('Nparticle_total_high', ('u4', 6)),
      ('flag_entropy', 'i4'),
      ('flag_double', 'i4'),
      ('flag_ic_info', 'i4'),
      ('flag_lpt_scalingfactor', 'i4'),
      ('unused', ('i4', 12)),
    ])
    self.header = self.file.read_record(dt, 1)[0]
    self.cellsize = self.header['cellsize']
    self.Nc = self.header['Ncells']
    self.Nc1 = self.header['Ncell1d']
    self.Np = self.file.read_record(dtype('u8'), self.Nc)
    self.Nf = self.file.read_record(dtype('u4'), self.Nc)
    self.cells = {}
    for i in range(self.Nc):
      self.cells[i] = self.file.read_record(dtype('u4'), self.Nf[i])
  def get_ids(self, corner1, corner2) :
    print self.cellsize
    ic1 = int32(floor(corner1 / self.cellsize))
    ic2 = int32(ceil(corner2 / self.cellsize))
    s = set([])
    print ic1, ic2
    for ix in range(ic1[0], ic2[0]):
      for iy in range(ic1[1], ic2[1]):
        for iz in range(ic1[2], ic2[2]):
          i = ix * self.Nc1 * self.Nc1 + iy * self.Nc1 + iz
          s = s.union(set(self.cells[i]))
    return list(s)
    
class Snapdir:
  def __init__(self, name_template, mapfile, reader):
    self.template = name_template
    self.reader = reader
    self.map = Snapmap(mapfile)
  def open(self, id):
    return Snapshot(self.template % id, self.reader) 

  def get_ids(self, corner1, corner2):
    return self.map.get_ids(corner1, corner2) 
