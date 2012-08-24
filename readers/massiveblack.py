class Reader:
    format = 'F'
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
      ('unused', ('f4', 12)),
    ]
    schema = [
      ('pos', ('f4', 3), [0, 1, 2, 3, 4, 5], []),
      ('vel', ('f4', 3), [0, 1, 2, 3, 4, 5], []),
      ('id', 'u8', [0, 1, 2, 3, 4, 5], []),
      ('mass', 'f4', [0, 2, 3, 4, 5], []),
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
    defaults = {
      'flag_sfr': 1,
      'flag_sft': 1,
      'flag_met': 1,
      'flag_entropy': 0,
      'flag_double': 0,
      'flag_ic_info': 0,
      'flag_cool': 1,
      'flag_feedback': 1,
    }
    class Constants:
      OmegaB = 0.044
      PhysDensThresh = 0.000831188
      @property
      def Ntot(self):
        return self.virtarray(6, 'u8', 
            lambda i: self['Ntot_low'][i] + (self['Ntot_high'].as_type('u8') << 32)[i],
            lambda i, value: ( self['Ntot_low'].__setitem__(i, value),
                               self['Ntot_high'].__setitem__(i, value >> 32))
            )
    constants = {
     'N': 'N',
     'boxsize': 'boxsize',
     'Ntot': (lambda h: h['Ntot_low'] + (h['Ntot_high'].astype('u8') << 32),
              lambda v: {'Ntot_low': v, 'Ntot_high': v >> 32}),
     'OmegaB' : 0.044,
     'PhysDensThresh': 0.000831188,
     'OmegaL': 'OmegaL',
     'OmegaM': 'OmegaM',
     'h': 'h',
     'redshift': 'redshift',
     'time': 'time',
     'Nfiles': 'Nfiles',
    }
