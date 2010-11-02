import gadget.reader
from gadget.io import CFile

class Reader(gadget.reader.Reader):
  def __init__(self) :
    gadget.reader.Reader.__init__(self, 
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
      ('ea', 'f4', [0], []),
      ('nha', 'f4', [0], []),
      ('sml', 'f4', [0], []),
      ('sfr', 'f4', [0], []),
      ('sft', 'f4', [4], []),
      ('met', 'f4', [0, 4], []),
      ('bhmass', 'f4', [5], []),
      ('bhmdot', 'f4', [5], []),
      ('type', 'u4', [0, 1, 4, 5], []),
      ('file', 'u4', [0, 1, 4, 5], []),
    ]
    );
  def constants(self, snapshot):
    h = snapshot.header
    return dict(
      OmegaB = 0.044,
      OmegaL = 0.0,
      OmegaM = 0.0,
      h = 0.72,
      N = h['N'],
      Z = 0,
      L = 0,
    )
