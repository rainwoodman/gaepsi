import _gadgetbase
class Reader:
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
    ]
    format = 'F'
    schema = [
      ('pos', ('f4', 3), [0, 1, 2, 3, 4, 5], []),
      ('vel', ('f4', 3), [0, 1, 2, 3, 4, 5], []),
      ('id', 'u4', [0, 1, 2, 3, 4, 5], []),
      ('mass', 'f4', [0, 1, 2, 3, 4, 5], []),
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
    ]
    class constants(_gadgetbase.constants):
      OmegaB = 0.044
      flag_sfr = 1
      flag_sft = 1
      flag_met = 1
      flag_feedback = 1
      flag_cool = 1
      flag_entropy = 0
      flag_double = 0
      flag_ic_info = 0
