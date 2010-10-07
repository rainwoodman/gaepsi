from gadget.schema import Schema
from gadget import schemas
from gadget.io import CFile
from numpy.ma import make_mask_none
from numpy.core import argsort
from matplotlib.mlab import find

class _Schema(Schema):

  def pad(self, original_size):
    if original_size != 0: return original_size
    else : return 4

  def update_meta(self, snapshot, header):
    Ndark = header['length'] - header['Ngas'] - header['Nstar'] - header['Nblack']
    snapshot.Nparticle = [ header['Ngas'], Ndark, 0, 0, header['Nstar'], header['Nblack']]
    snapshot.C['OmegaB'] = 0.04
    snapshot.C['OmegaM'] = NaN # figure this out later
    snapshot.C['OmegaL'] = NaN # figure this out later
    snapshot.C['h'] = NaN  # figure this out later
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

schemas['D6subgroup'] = _Schema([
      ('pos', ('f4', 3), [0, 1, 4, 5]),
      ('vel', ('f4', 3), [0, 1, 4, 5]),
      ('id', 'u4', [0, 1, 4, 5]),
      ('mass', 'f4', [0, 4, 5]),
      ('ie', 'f4', [0]),
      ('rho', 'f4', [0]),
      ('ea', 'f4', [0]),
      ('nha', 'f4', [0]),
      ('sml', 'f4', [0]),
      ('sfr', 'f4', [0]),
      ('sft', 'f4', [4]),
      ('met', 'f4', [0, 4]),
      ('type', 'u4', [0, 1, 4, 5]),
    ],
    [ ('id', 'i4'),
      ('length', 'i4'),
      ('Ngas', 'i4'),
      ('Nstar', 'i4'),
      ('Nblack', 'i4')
    ], padding = None
    )
