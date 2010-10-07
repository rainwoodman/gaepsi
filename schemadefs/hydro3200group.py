from gadget.schema import Schema
from gadget import schemas
from gadget.io import CFile
from numpy.ma import make_mask_none
from numpy.core import argsort
from numpy.core import array
from numpy.core import dtype
from matplotlib.mlab import find

class _Schema(Schema):

  def pad(self, original_size):
    return original_size

  def update_meta(self, snapshot, header):
    snapshot.Nparticle = array(header['Nparticle'], dtype=dtype('i8'))
    snapshot.C['OmegaB'] = 0.044
    snapshot.C['OmegaM'] = 0.26
    snapshot.C['OmegaL'] = 0.74
    snapshot.C['h'] = 0.72
    snapshot.C['H'] = 0.1
    snapshot.C['G'] = 43007.1

  def open(self, snapshot, fname):
    if hasattr(fname, 'readline') :
      snapshot.file = CFile(fname);
    else :
      snapshot.file = CFile(fname, 'r')

  def post_init(self, snapshot):
    snapshot.file.seek(snapshot.offsets['type'])
    blockschema = self.__dict['type']
    length = snapshot.sizes['type'] / blockschema.dtype.itemsize
    snapshot.type_index = snapshot.file.read_record(blockschema.dtype, length)

  def reindex(self, snapshot, name):
    blockschema = self.__dict[name]
    mask = make_mask_none(snapshot.type_index.shape)
    for ptype in blockschema.ptypes:
      mask = mask | (snapshot.type_index == ptype)
    sortind = argsort(snapshot.type_index[find(mask)], kind="mergesort")
    
    snapshot.D[name] = snapshot.D[name][sortind]
    Schema.reindex(self, snapshot, name)

schemas['hydro3200group'] = _Schema([
      ('pos', ('f4', 3), [0, 1, 4, 5]),
      ('vel', ('f4', 3), [0, 1, 4, 5]),
      ('id', 'u8', [0, 1, 4, 5]),
      ('mass', 'f4', [0, 4, 5]),
      ('ie', 'f4', [0]),
      ('rho', 'f4', [0]),
      ('ea', 'f4', [0]),
      ('nha', 'f4', [0]),
      ('sml', 'f4', [0]),
      ('sfr', 'f4', [0]),
      ('sft', 'f4', [4]),
      ('met', 'f4', [0, 4]),
      ('bhmass', 'f4', [5]),
      ('bhmdot', 'f4', [5]),
      ('bhnprogs', 'u8', [5]),
      ('type', 'u4', [0, 1, 4, 5]),
      ('file', 'u4', [0, 1, 4, 5]),
    ],
    [
      ('Nparticle', ('u8', 6)),
      ('expectedNparticle', ('u8', 6)),
      ('totmass', 'f4'),
      ('pos', ('f4', 3)),
      ('vel', ('f4', 3))
    ], padding = None
    )
