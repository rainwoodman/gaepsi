def Snapshot(idtype='i8', floattype='f4', 
       blocks=['pos', 'vel', 'id', 'mass', 'ie'],
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
      ('flag_double', 'i4'),
      ('flag_ic_info', 'i4'),
      ('flag_lpt_scalingfactor', 'i4'),
      ('unused', ('u4', 12)),
    ]
    class schema:
      pos = (floattype, 3), [0, 1, 2, 3, 4, 5]
      vel = (floattype, 3), [0, 1, 2, 3, 4, 5]
      id = idtype, [0, 1, 2, 3, 4, 5]
      mass = floattype, [0, 1, 2, 3, 4, 5]
      ie = floattype, [0]
      rho = floattype, [0]
      ye = floattype, [0], ['flag_cool']
      xHI = floattype, [0], ['flag_cool']
      sml = floattype, [0]
      sfr = floattype, [0], ['flag_sfr']
      sft = floattype, [4], ['flag_sft']
      met = floattype, [0, 4], ['flag_met']
      bhmass = floattype, [5]
      bhmdot = floattype, [5]
      bhnprogs = 'i8', [5]
      __blocks__ = blocks
      
    class constants:
      def _getNtot(self, i):
        return self['Ntot_low'][i] + (int(self['Ntot_high'][i]) << 32)
      def _setNtot(self, i, value):
        self['Ntot_low'][i] = value & ((1 << 32) - 1)
        self['Ntot_high'][i] = (value >> 32)
      Ntot = (('i8', 6), _getNtot, _setNtot)
  return Reader
