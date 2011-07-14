from gaepsi.readers import ReaderBase, F77File

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

class Reader(ReaderBase):
  def __init__(self) :
    ReaderBase.__init__(self, 
    F77File, 
    header = header,
    schemas = [
      ('pos', ('f4', 3), [0, 1, 4, 5], []),
      ('vel', ('f4', 3), [0, 1, 4, 5], []),
      ('id', 'u4', [0, 1, 4, 5], []),
      ('mass', 'f4', [0, 4, 5], []),
      ('ie', 'f4', [0], []),
      ('rho', 'f4', [0], []),
      ('ye', 'f4', [0], []),
      ('xHI', 'f4', [0], []),
      ('sml', 'f4', [0], []),
      ('sfr', 'f4', [0], ['flag_sfr']),
      ('sft', 'f4', [4], ['flag_sft']),
      ('met', 'f4', [0, 4], ['flag_met']),
      ('bhmass', 'f4', [5], []),
      ('bhmdot', 'f4', [5], [])
    ],
    constants = {
     'OmegaB' : 0.044,
     'PhysDensThresh': 0.000831188,
     'OmegaL': 'OmegaL',
     'OmegaM': 'OmegaM',
     'h': 'h',
     'N': 'N',
     'Z': 'redshift',
     'L': 'boxsize',
    }
    );
