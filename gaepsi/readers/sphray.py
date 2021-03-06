import _gadgetbase

class Snapshot:
    format = 'F'
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
    ]
    class schema:
      pos = ('f4', 3), [0], []
      vel = ('f4', 3), [0], []
      id = 'u4', [0], []
      mass = 'f4', [0], []
      ie = 'f4', [0], []
      rho = 'f4', [0], []
      ye = 'f4', [0], []
      xHI = 'f4', [0], []
      sml = 'f4', [0], []
      T = 'f4', [0], []
      Hmf = 'f4', [0], ['flag_Hmf']
      Hemf = 'f4', [0], ['flag_Hemf']
      HeIa = 'f4', [0], ['flag_Helium']
      HeIIa = 'f4', [0], ['flag_Helium']
      gammaHI = 'f4', [0], ['flag_gammaHI']
      HIa_cloudy = 'f4', [0], ['flag_cloudy']
      eos = 'f4', [0], ['flag_eos']
      lasthit = 'u8', [0], []
      __blocks__ = ["pos", "vel", "id", "mass", "ie", 
                    "rho", "ye", "xHI", "sml", "T", "Hmf", 
                    "Hemf", "HeIa", "HeIIa", "gammaHI", "HIa_cloudy", 
                    "eos", "lasthit" ]
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
      flag_gammaHI = 0
      flag_cloudy = 0
      flag_eos = 0
      def _getNtot(self, i):
        return self['Ntot_low'][i] + (int(self['Ntot_high'][i]) << 32)
      def _setNtot(self, i, value):
        self['Ntot_low'][i] = value
        self['Ntot_high'][i] = (value >> 32)
      Ntot = (('i8', 6), _getNtot, _setNtot)


