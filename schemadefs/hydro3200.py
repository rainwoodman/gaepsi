from gadget.schema import Schema
from gadget import schemas
from gadget.io import F77File

class _Schema(Schema):

  def pad(self, original_size):
    if original_size != 0: return original_size + 8
    else : return 0

  def update_meta(self, snapshot, header):
    snapshot.Nparticle = header['Nparticle']
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
    
schemas['hydro3200'] = _Schema([
      ('pos', ('f4', 3), [0, 1, 4, 5], []),
      ('vel', ('f4', 3), [0, 1, 4, 5], []),
      ('id', 'u8', [0, 1, 4, 5], []),
      ('mass', 'f4', [0, 4, 5], []),
      ('ie', 'f4', [0], []),
      ('rho', 'f4', [0], []),
      ('ea', 'f4', [0], []),
      ('nha', 'f4', [0], []),
      ('sml', 'f4', [0], []),
      ('sfr', 'f4', [0], ['flag_sfr']),
      ('sft', 'f4', [4], ['flag_sft']),
      ('met', 'f4', [0, 4], ['flag_met']),
      ('bhmass', 'f4', [5], []),
      ('bhmdot', 'f4', [5], [])
    ],
    [
      ('Nparticle', ('u4', 6)),
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
  ]);
