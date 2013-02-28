from collections import OrderedDict

def Snapshot(idtype, floattype, constants, blocks=None):
  baseconstants=constants
  if blocks is None:
    blocks = ['pos', 'vel', 'id', 'mass', 'ie', 'rho', 'ye', 
            'xHI', 'sml', 'sfr', 'sft', 'met', 'bhmass', 'bhmdot', 'bhnprogs']
  class MyReader:
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
      
    class constants(baseconstants):
      def _getNtot(self, i):
        return self['Ntot_low'][i] + (int(self['Ntot_high'][i]) << 32)
      def _setNtot(self, i, value):
        self['Ntot_low'][i] = value
        self['Ntot_high'][i] = (value >> 32)
      Ntot = (('i8', 6), _getNtot, _setNtot)

  class MyGroupTab:
      format = 'C'
      usemasstab = False
      header = [('N', ('i4', (1,))),
                ('Ntot', ('i4', (1,))),
                ('Nids',  'i4'),
                ('TotNids', 'u8'),
                ('Nfiles', 'i4')]
      class schema:
        length = 'i4', 0
        offset = 'i4', 0
        mass = 'f4', 0
        pos = ('f4', 3), 0
        vel = ('f4', 3), 0
        lenbytype = ('u4', 6), 0
        massbytype = ('f4', 6), 0
        sfr = 'f4', 0
        bhmass = 'f4', 0
        bhmdot = 'f4', 0
        __blocks__ = ['length', 'offset', 'mass', 'pos', 'vel', 
             'lenbytype', 'massbytype', 'sfr', 'bhmass', 'bhmdot']
      class constants(baseconstants): pass
    
  class MySubHaloTab:
      format = 'C'
      usemasstab = False
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
        haloid = idtype, 1
        halogroup = 'u4', 1
    
        __blocks__ = [
        'length', 'offset', 'mass', 'pos', 
        'mmean200', 'rmean200', 
        'mcrit200', 'rcrit200', 
        'mtoph200', 'rtoph200', 
        'veldispmean200', 'veldispcrit200', 'veldisptoph200', 
        'lencontam', 'masscontam', 
        'nhalo', 'firsthalo', 
        'halolen', 'halooffset', 'haloparent', 
        'halomass', 'halopos', 'halovel', 'halocm', 
        'halospin', 'haloveldisp', 'halovmax', 'halovmaxrad', 
        'halohalfmassradius', 
        'haloid', 'halogroup',
        ]
    
      class constants:
        def _getN(self, i):
          return [self['Ngroups'], self['Nsubgroups']][i]
        def _setN(self, i, value):
          if i == 0: self['Ngroups'] = value
          if i == 1: self['Nsubgroups'] = value
        N = (('i8', 2), _getN, _setN)

  return MyReader, MyGroupTab, MySubHaloTab
