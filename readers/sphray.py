import gadget.reader
from gadget.io import F77File

class Reader(gadget.reader.Reader):
  def __init__(self) :
    gadget.reader.Reader.__init__(self, 
    F77File, 
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
      ('OmegaB', 'f8'),
      ('nrays', 'u8'),
      ('flag_Hmf', 'i4'),
      ('flag_Hemf', 'i4'),
      ('flag_Helium', 'i4'),
      ('flag_gammaHI', 'i4'),
      ('flag_cloudy', 'i4'),
      ('flag_eos', 'i4'),
      ('unused', ('i4', 5)),
    ],
    schemas = [
      ('pos', ('f4', 3), [0], []),
      ('vel', ('f4', 3), [0], []),
      ('id', 'u4', [0], []),
      ('mass', 'f4', [0], []),
      ('ie', 'f4', [0], []),
      ('rho', 'f4', [0], []),
      ('ea', 'f4', [0], []),
      ('nha', 'f4', [0], []),
      ('sml', 'f4', [0], []),
      ('T', 'f4', [0], []),
      ('Hmf', 'f4', [0], ['flag_Hmf']),
      ('Hemf', 'f4', [0], ['flag_Hemf']),
      ('HeIa', 'f4', [0], ['flag_Helium']),
      ('HeIIa', 'f4', [0], ['flag_Helium']),
      ('gammaHI', 'f4', [0], ['flag_gammaHI']),
      ('HIa_cloudy', 'f4', [0], ['flag_cloudy']),
      ('eos', 'f4', [0], ['flag_eos']),
      ('lasthit', 'u8', [0], []),
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


