def Snapshot(idtype='u8', floattype='f4', 
       blocks=['pos', 'vel', 'id', 'mass', 'ie', 'entropy', 'rho', 'rhoegy', 'ye', 
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
      ('flag_double', 'i4'),
      ('flag_ic_info', 'i4'),
      ('flag_lpt_scalingfactor', 'i4'),
      ('flag_pressure_entropy', 'i1'),
      ('Ndims', 'i1'),
      ('densitykerneltype', 'i1'),
      ('unused', ('u1', 45)),
    ]
    class schema:
      pos = (floattype, 3), [0, 1, 2, 3, 4, 5]
      vel = (floattype, 3), [0, 1, 2, 3, 4, 5]
      id = idtype, [0, 1, 2, 3, 4, 5]
      mass = floattype, [0, 1, 2, 3, 4, 5]
      ie = floattype, [0]
      entropy = floattype, [0], ['flag_pressure_entropy']
      rho = floattype, [0]
      rhoegy = floattype, [0], ['flag_pressure_entropy']
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

def GroupTab(idtype='u8', floattype='f4', **kwargs):
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
        mass = floattype, 0
        pos = (floattype, 3), 0
        vel = (floattype, 3), 0
        lenbytype = ('u4', 6), 0
        massbytype = (floattype, 6), 0
        sfr = floattype, 0
        bhmass = floattype, 0
        bhmdot = floattype, 0
        __blocks__ = ['length', 'offset', 'mass', 'pos', 'vel', 
             'lenbytype', 'massbytype', 'sfr', 'bhmass', 'bhmdot']
      class constants: 
        pass
  return MyGroupTab

def GroupIDs(idtype='u8', floattype='f4', **kwargs):
  class MyGroupIDs:
      format = 'C'
      usemasstab = False
      header = [('Nhalo', 'i4'),
                ('Nhalotot', 'i4'),
                ('N',  ('i4', (1, ))),
                ('Ntot', ('i8', (1, ))),
                ('Nfiles', 'i4'),
                ('junk', 'i4'),
                ]
      class schema:
        id = idtype, 0
        __blocks__ = ['id']
      class constants: 
        pass
  return MyGroupIDs
    
def SubHaloTab(idtype='i8', floattype='f4', **kwargs):
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
        length = 'i4',  0     # number of partiles in group
        offset = 'i4',  0     # do not use. offset of first particle in
                              # concatenated ids file(overflown)
        mass = floattype,    0 # mass of group
        pos = (floattype, 3), 0  # center mass postion of group
        mmean200 = floattype, 0  # mass up to 200 mean density?
        rmean200 = floattype, 0  # radius up to 200 mean density?
        mcrit200 = floattype, 0  #                 critical?
        rcrit200 = floattype, 0  #                 critical?
        mtoph200 = floattype, 0  #                 tophat
        rtoph200 = floattype, 0  #                 tophat
        veldispmean200 = floattype, 0 # velocity dispersion, for each
                                      #  overdensity 200
        veldispcrit200 = floattype, 0
        veldisptoph200 = floattype, 0
        lencontam = 'i4', 0       # number of particles (contamination) not in
                                  # any subhalos.
        masscontam = floattype, 0 # mass of particles not in any subhalos
        nhalo = 'i4', 0           # number of subhalos
        firsthalo = 'i4', 0       # offset of first halo if you concatenate all
                                  # subhalos in all tab files.
        halolen = 'i4', 1         # number of particles in this subhalo
        halooffset = 'i4', 1      # do not use. offset of first particle in
                                  # concatenated ids file(overflown)
        haloparent = 'i4', 1      # index of parent (don't know what it is)
        halomass = floattype, 1   # total mass of subhalo
        halopos = (floattype, 3), 1  # potential center position
        halovel = (floattype, 3), 1  # velocity (center mass?)
        halocm = (floattype, 3), 1   # center of mass position
        halospin = (floattype, 3), 1  # spin
        haloveldisp = floattype, 1  # velocity dispersion
        halovmax = floattype, 1    # max circular velocity
        halovmaxrad = floattype, 1  # radius of max circular velocity
        halohalfmassradius = floattype, 1 # radius of half the mass
        haloid = idtype, 1         # id of most bound particle
        halogroup = 'u4', 1        # index of group containing this subhalo
    
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
  return MySubHaloTab 
