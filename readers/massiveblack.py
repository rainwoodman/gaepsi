import _gadgetbase
class constants:
      OmegaB = 0.044
      PhysDensThresh = 0.000831188
      flag_sfr = 1
      flag_sft = 1
      flag_met = 1
      flag_feedback = 1
      flag_cool = 1
      flag_entropy = 0
      flag_double = 0
      flag_ic_info = 0

Snapshot = _gadgetbase.Snapshot(
     idtype='u8',
     floattype='f4',
     constants=constants)

class GroupTab:
  format = 'C'
  header = [('N', ('i4', (1,))),
            ('Ntot', ('i4', (1,))),
            ('Nids',  'i4'),
            ('TotNids', 'u8'),
            ('Nfiles', 'i4')]
  class schema:
    length = 'i4'
    offset = 'i4'
    mass = 'f4'
    pos = ('f4', 3)
    vel = ('f4', 3)
    lenbytype = ('u4', 6)
    massbytype = ('f4', 6)
    sfr = 'f4'
    bhmass = 'f4'
    bhmdot = 'f4'
    __blocks__ = ['length', 'offset', 'mass', 'pos', 'vel', 
         'lenbytype', 'massbytype', 'sfr', 'bhmass', 'bhmdot']

class SubHaloTab:
  format = 'C'
  header = [('Ngroups', ('i4', (1,))),
            ('NtotGroups', ('i4', (1,))),
            ('Nids',  'i4'),
            ('TotNids', 'u8'),
            ('Nfiles', 'i4'),
            ('Nsubgroups', ('i4', (1,))),
            ('NtotSubgroups', ('i4', (1,))),]
  class schema:
    length = 'i4',  0
    offset = 'i4',  0
    mass = 'f4',    0
    pos = ('f4', 3), 0
    mmean200 = 'f4', 0
    rmean200 = 'f4', 0
    mcrit200 = 'f4', 0
    rcrit200 = 'f4', 0
    mtoph200 = 'f4', 0
    rtoph200 = 'f4', 0
    veldispmean200 = 'f4', 0
    veldispcrit200 = 'f4', 0
    veldisptoph200 = 'f4', 0
    lencontam = 'i4', 0
    masscontam = 'f4', 0
    nhalo = 'i4', 0
    firsthalo = 'i4', 0
    halolen = 'i4', 1
    halooffset = 'i4', 1
    haloparent = 'i4', 1
    halomass = 'f4', 1
    halopos = ('f4', 3), 1
    halovel = ('f4', 3), 1
    halocm = ('f4', 3), 1
    halospin = ('f4', 3), 1
    haloveldisp = 'f4', 1
    halovmax = 'f4', 1
    halovmaxrad = 'f4', 1
    halohalfmassradius = 'f4', 1
    haloid = 'u8', 1
    halogroup = 'u4', 1

    __blocks__ = ['length', 'offset', 'mass', 'pos', 'vel', 
         'lenbytype', 'massbytype', 'sfr', 'bhmass', 'bhmdot']
  class constants:
    def _getN(self, i):
      return [self['Ngroups'], self['Nsubgroups']][i]
    def _setN(self, i, value):
      if i == 0: self['Ngroups'] = value
      if i == 1: self['Nsubgroups'] = value
    N = (('i8', 2), _getN, _setN)
