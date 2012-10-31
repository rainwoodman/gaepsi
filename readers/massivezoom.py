from massiveblack import Reader as Base
class Reader(Base):
    schema = [
      ('pos', ('f8', 3), [0, 1, 2, 3, 4, 5], []),
      ('vel', ('f8', 3), [0, 1, 2, 3, 4, 5], []),
      ('id', 'u8', [0, 1, 2, 3, 4, 5], []),
      ('mass', 'f8', [0, 1, 2, 3, 4, 5], []),
      ('ie', 'f8', [0], []),
      ('rho', 'f8', [0], []),
      ('ye', 'f8', [0], []),
      ('xHI', 'f8', [0], []),
      ('sml', 'f8', [0], []),
      ('sfr', 'f8', [0], ['flag_sfr']),
      ('sft', 'f8', [4], ['flag_sft']),
      ('met', 'f8', [0, 4], ['flag_met']),
      ('bhmass', 'f8', [5], []),
      ('bhmdot', 'f8', [5], [])
    ]
