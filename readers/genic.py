from gaepsi.readers import ReaderBase, F77File

header = [
      ('N', ('u4', 6)),
      ('mass', ('f8', 6)),
      ('time', 'f8'),
      ('redshift', 'f8'),
      ('flag_sfr', 'i4'),
      ('flag_feedback', 'i4'),
      ('Ntot_low', ('u4', 6)),
      ('flag_cool', 'i4'),
      ('Nfiles', 'i4'),
      ('boxsize', 'f8'),
      ('OmegaM', 'f8'),
      ('OmegaL', 'f8'),
      ('h', 'f8'),
      ('flag_sft', 'i4'),
      ('flag_met', 'i4'),
      ('Ntot_high', ('u4', 6)),
      ('flag_entropy', 'i4'),
      ('flag_double', 'i4'),
      ('flag_ic_info', 'i4'),
      ('flag_lpt_scalingfactor', 'i4'),
      ('unused', ('i4', 12)),
    ];

class Reader(ReaderBase):
  def __init__(self) :
    ReaderBase.__init__(self, 
    F77File, 
    header = header,
    schemas = [
      ('pos', ('f4', 3), [0,1,2,3,4,5], []),
      ('vel', ('f4', 3), [0,1,2,3,4,5], []),
      ('id', 'u8', [0,1,2,3, 4,5], []),
      ('mass', 'f4', [2,3,4,5], []),
      ('ie', 'f4', [0], []),
    ],
    defaults = {
      'flag_sfr': 0,
      'flag_sft': 0,
      'flag_met': 0,
      'flag_entropy': 0,
      'flag_double': 0,
      'flag_ic_info': 0,
      'flag_cool': 0,
      'flag_feedback': 0,
    },
    constants = {
     'N': 'N',
     'Ntot': (lambda h: h['Ntot_low'] + (h['Ntot_high'].astype('u8') << 32),
              lambda v: {'Ntot_low': v, 'Ntot_high': v >> 32}),
     'OmegaB' : 0.044,
     'OmegaL': 'OmegaL',
     'OmegaM': 'OmegaM',
     'h': 'h',
     'redshift': 'redshift',
     'time': 'time',
     'boxsize': 'boxsize',
     'Nfiles': 'Nfiles',
    }
    );
