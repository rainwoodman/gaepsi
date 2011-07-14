from gaepsi.readers import ReaderBase, CFile

class Reader(ReaderBase):
  def __init__(self) :
    ReaderBase.__init__(self, 
    CFile, 
    header = [
      ('N', ('u8', 6)),
      ('Nex', ('u8', 6)),
      ('totmass', 'f4'),
      ('pos', ('f4', 3)),
      ('vel', ('f4', 3)),
    ],
    schemas = [
      ('pos', ('f4', 3), [0, 1, 4, 5], []),
      ('vel', ('f4', 3), [0, 1, 4, 5], []),
      ('id', 'u8', [0, 1, 4, 5], []),
      ('mass', 'f4', [0, 4, 5], []),
      ('ie', 'f4', [0], []),
      ('rho', 'f4', [0], []),
      ('ye', 'f4', [0], []),
      ('xHI', 'f4', [0], []),
      ('sml', 'f4', [0], []),
      ('sfr', 'f4', [0], []),
      ('sft', 'f4', [4], []),
      ('met', 'f4', [0, 4], []),
      ('bhmass', 'f4', [5], []),
      ('bhmdot', 'f4', [5], []),
      ('type', 'u4', [0, 1, 4, 5], []),
      ('file', 'u4', [0, 1, 4, 5], []),
    ],
    constants = {
     'OmegaB' : 0.044,
     'PhysDensThresh': 0.000831188,
     'OmegaL': 'OmegaL',
     'OmegaM': 'OmegaM',
     'h': 'h',
     'N': 'N',
     'Z': 'redshift',
     'L': 'boxsize',
    }
    );
