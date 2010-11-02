import gadget.reader
from gadget.io import F77File
from numpy import array
header = [
      ('N', 'u8'),
      ('Nids', 'u8'),
      ('unused0', ('u4',2)),
      ('mass', ('f8', 6)),
      ('time', 'f8'),
      ('redshift', 'f8'),
      ('flag_sfr', 'i4'),
      ('flag_feedback', 'i4'),
      ('TotNgroups', 'u8', 1),
      ('TotNids', 'u8', 1),
      ('unused1', ('u4', 2)),
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

class Reader(gadget.reader.Reader):
  def __init__(self) :
    gadget.reader.Reader.__init__(self, 
    F77File, 
    header = header,
    schemas = [
      ('length', 'u8', [0], []),
      ('offset', 'u8', [0], []),
      ('totmass', 'f4', [0], []),
      ('pos', ('f4', 3), [0], []),
      ('vel', ('f4', 3), [0], []),
      ('N', ('u4', 6), [0], []),
      ('mass', ('f4', 6), [0], []),
      ('sfr', 'f4', [0], []),
      ('bhmass', 'f4', [0], []),
      ('bhmdot', 'f4', [0], [])
    ]
    );
  def constants(self, snapshot):
    h = snapshot.header
    return dict(
      OmegaB = 0.044,
      OmegaL = h['OmegaL'],
      OmegaM = h['OmegaM'],
      h = h['h'],
      N = array([h['N'], 0, 0, 0, 0, 0], dtype='u4'),
      Z = h['redshift'],
      L = h['boxsize'],
    )
