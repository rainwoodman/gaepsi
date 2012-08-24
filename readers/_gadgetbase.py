class constants:
   def _getNtot(self, i):
     return self['Ntot_low'][i] + (int(self['Ntot_high'][i]) << 32)
   def _setNtot(self, i, value):
     self['Ntot_low'][i] = value
     self['Ntot_high'][i] = (value >> 32)
   Ntot = (('u8', 6), _getNtot, _setNtot)
