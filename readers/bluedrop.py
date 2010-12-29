from reader import Reader as Base
from io import F77File

header = [
      ('N', ('u4', 6)),
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
    ];

class Reader(Base):
  def __init__(self) :
    Base.__init__(self, 
    F77File, 
    endian = '>', 
    header = header,
    schemas = [
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
    ]
    );
  def constants(self, snapshot):
    h = snapshot.header
    return dict(
      OmegaB = 0.044,
      OmegaL = h['OmegaL'],
      OmegaM = h['OmegaM'],
      h = h['h'],
      N = h['N'],
      Z = h['redshift'],
      L = h['boxsize'],
    )
