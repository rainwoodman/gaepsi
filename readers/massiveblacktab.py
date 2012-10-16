class Reader:
  format = 'C'
  header = [('Ngroups', ('i4', (1,))),
            ('TotNgroups', ('i4', (1,))),
            ('Nids',  'i4'),
            ('TotNids', 'u8'),
            ('Nfiles', 'i4')]
  schema = [('length', 'i4', (0, )),
            ('offset', 'i4', (0, )),
            ('mass', 'f4',   (0, )),
            ('pos', ('f4', 3),(0, )),
            ('vel', ('f4', 3),(0, )),
            ('lenbytype', ('u4', 6), (0, )),
            ('massbytype', ('f4', 6), (0, )),
            ('sfr', 'f4', (0, )),
            ('bhmass', 'f4', (0, )),
            ('bhmdot', 'f4', (0, )),
            ]
  class constants:
    N = 'Ngroups'
    Ntot = 'TotNgroups'
    Nfiles = 'Nfiles'
