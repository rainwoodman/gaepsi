from gadget.schema import Schema
from gadget import schemas
from gadget.io import F77File
from numpy import int64
from numpy import array
from numpy import dtype

class _Schema(Schema):
  def pad(self, original_size):
    if original_size != 0: return original_size + 8
    else : return 0

  def update_meta(self, snapshot, header):
    snapshot.Nparticle = array([header['Ngroups'], 0, 0, 0, 0, 0,  ], dtype('i8'))
    snapshot.C['OmegaB'] = 0.044
    snapshot.C['OmegaM'] = 0.26
    snapshot.C['OmegaL'] = 0.74
    snapshot.C['h'] = 0.72
    snapshot.C['H'] = 0.1
    snapshot.C['G'] = 43007.1

  def open(self, snapshot, fname):
    if hasattr(fname, 'readline') :
      snapshot.file = F77File(fname);
    else :
      snapshot.file = F77File(fname, 'r')
    
schemas['hydro3200tab'] = _Schema([
      ('length', 'u8', [0]),
      ('offset', 'u8', [0]),
      ('totmass', 'f4', [0]),
      ('pos', ('f4', 3), [0]),
      ('vel', ('f4', 3), [0]),
      ('N', ('u4', 6), [0]),
      ('mass', ('f4', 6), [0]),
      ('sfr', 'f4', [0]),
      ('bhmass', 'f4', [0]),
      ('bhmdot', 'f4', [0])
    ],
    [
      ('Ngroups', ('u8', 1)),
      ('Nids', ('u8', 1)),
      ('unused0', ('u4', 2)),
      ('mass', ('f8', 6)),
      ('time', 'f8'),
      ('redshift', 'f8'),
      ('flag_sfr', 'i4'),
      ('flag_feedback', 'i4'),
      ('TotNgroups', ('u8', 1)),
      ('TotNids', ('u8', 1)),
      ('unused1', ('u4', 2)),
      ('flag_cool', 'i4'),
      ('Nfiles', 'i4'),
      ('boxsize', 'f8'),
      ('OmegaM', 'f8'),
      ('OmegaL', 'f8'),
      ('h', 'f8'),
      ('flag_sft', 'i4'),
      ('flag_met', 'i4'),
      ('unused2', ('u4', 6)),
      ('flag_entropy', 'i4'),
      ('flag_double', 'i4'),
      ('flag_ic_info', 'i4'),
      ('flag_lpt_scalingfactor', 'i4'),
#      ('unused', ('i4', 12)),
  ]);
