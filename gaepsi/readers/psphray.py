def Snapshot(idtype='i8', floattype='f4', 
       blocks=['pos', 'vel', 'id', 'mass', 'ie', 'rho', 'ye', 
            'xHI', 'sml', 'sfr', 'sft', 'met', 
            'bhmass', 'bhmdot', 'bhnprogs'],
        **kwargs
       ):
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
      ('OmegaB', 'f8'),
      ('nrays', 'u8'),
      ('flag_Hmf', 'i4'),
      ('flag_Hemf', 'i4'),
      ('flag_Helium', 'i4'),
      ('flag_dep', 'i4'),
      ('flag_rec', 'i4'),
      ('flag_cloudy', 'i4'),
      ('flag_eos', 'i4'),
      ('unused', ('i4', 4)),
    ]
    class schema:
      pos = ('f4', 3), [0,4,5], []
      id = 'u8', [0,4,5], []
      mass = 'f4', [0,4,5], []
      ie = 'f4', [0], []
      rho = 'f4', [0], []
      ye = 'f4', [0], []
      xHI = 'f4', [0], []
      sml = 'f4', [0], []
      Hmf = 'f4', [0], ['flag_Hmf']
      Hemf = 'f4', [0], ['flag_Hemf']
      xHeI = 'f4', [0], ['flag_Helium']
      xHeII = 'f4', [0], ['flag_Helium']
      yGdepHI = 'f4', [0], ['flag_dep']
#      yGdepHeI = 'f4', [0], ['flag_dep']
#      yGdepHeII = 'f4', [0], ['flag_dep']
#      yGrecHII = 'f4', [0], ['flag_rec']
#      yGrecHeII = 'f4', [0], ['flag_rec']
#      yGrecHeIII = 'f4', [0], ['flag_rec']
#      HIa_cloudy = 'f4', [0], ['flag_cloudy']
#      eos = 'f4', [0], ['flag_eos']
      lasthit = 'i8', [0], []
      hits = 'i4', [0], []
      sft = 'f4', [4], []
      ngammas = 'f8', [5], []
      spec = 'u8', [5], []
      __blocks__ = ["pos", "id", "mass", "ie", 
                    "rho", "ye", "xHI", "sml", "Hmf",  
                    "Hemf", "xHeI", "xHeII", "yGdepHI", "lasthit", 
                    "hits", "sft", "ngammas", "spec" ]
      
    class constants:
      flag_sfr = 1
      flag_sft = 1
      flag_met = 1
      flag_entropy = 0
      flag_cool = 1
      flag_feedback = 1
      flag_Hmf = 0
      flag_Hemf = 0
      flag_Helium = 0
      flag_dep = 0
      flag_rec = 0
      flag_cloudy = 0
      flag_eos = 0
      def _getNtot(self, i):
        return self['Ntot_low'][i] + (int(self['Ntot_high'][i]) << 32)
      def _setNtot(self, i, value):
        self['Ntot_low'][i] = value & ((1 << 32) - 1)
        self['Ntot_high'][i] = (value >> 32)
      Ntot = (('i8', 6), _getNtot, _setNtot)
  return Reader
