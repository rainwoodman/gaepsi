import numpy
from gaepsi.compiledbase import fillingcurve
from gaepsi.compiledbase.ztree import Tree
from gaepsi.snapshot import Snapshot
from gaepsi.tools import packarray
from gaepsi.readers import F77File
from gaepsi.compiledbase.camera import Camera

class MeshIndex:
  headerdtype = numpy.dtype(
    [ ('origin', ('f8', 3)),
      ('boxsize', ('f8', 3)),
      ('bitsperaxis', 'i8'),
    ])
  def __init__(self, origin, boxsize, bitsperaxis, index):
    self.header = numpy.empty(None, dtype=MeshIndex.headerdtype)
    self.header['bitsperaxis'] = bitsperaxis
    self.header['origin'] = origin
    self.header['boxsize'] = boxsize
    self.index = index
    zkey = numpy.empty(1<<bitsperaxis * 3, dtype=fillingcurve.fckeytype) 
    zkey[:] = numpy.arange(1 << (bitsperaxis * 3))
    zkey[:] <<= (3 * fillingcurve.bits - bitsperaxis * 3)
    self.scale = fillingcurve.scale(self.header['origin'], self.header['boxsize'])
    self.tree = Tree(zkey, self.scale, maxthresh=1, minthresh=1)

  @classmethod
  def fromfile(cls, filename):
    with F77File(filename, mode='r') as file:
      header = file.read_record(cls.headerdtype)[0]
      start = file.read_record('i8')
      end = file.read_record('i8')
      A = file.read_record('u4')
      index = packarray(A, start, end)
      return MeshIndex(header['origin'],
                       header['boxsize'],
                       header['bitsperaxis'], 
                       index)

  def tofile(self, filename):
    with F77File(filename, mode='w') as file:
      file.write_record(self.header)
      file.write_record(numpy.int64(self.index.start))
      file.write_record(numpy.int64(self.index.end))
      file.write_record(numpy.int32(self.index.A))

  def cut(self, origin, boxsize):
    _origin = numpy.empty(3, dtype='f8')
    _boxsize = numpy.empty(3, dtype='f8')
    _origin[:] = origin
    _boxsize[:] = boxsize
    camera = Camera(1, 1)
    camera.lookat(target=_origin + _boxsize * 0.5,
           pos=_origin + _boxsize * 0.5 - [0, 0, _boxsize[2]], up=[0, 1, 0])
    camera.ortho(near=_boxsize[2]*0.5, far=_boxsize[2]*1.5,
           extent=(-_boxsize[0] * 0.5, _boxsize[0] * 0.5,
                   -_boxsize[1] * 0.5, _boxsize[1] * 0.5))
    mask = camera.prunetree(self.tree, return_nodes=False)
    return numpy.unique(self.index.compress(mask).A)
    
