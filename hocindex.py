from numpy import zeros
from numpy import unique
from scipy.weave import inline

class HOCIndex :
  def __init__(self, field):
    self.hoc = {}
    self.__build(field)

  def __del__(self):
    print 'deleting a hoc object'
    del self.hoc
    print 'deleted a hoc object'

  def get(self, x, y):
    cellsize = self.cellsize
    ncells = self.ncells
    xcc = int(x / cellsize)
    ycc = int(y / cellsize)

    xc = self.__wrap(xcc)
    yc = self.__wrap(ycc)
    return self.hoc[xc * ncells + yc]

  def quickimage(self) :
    image = zeros((self.ncells, self.ncells))
    for i in range(self.ncells) :
      for j in range(self.ncells) :
        image[i,j] = self.hoc[i * self.ncells + j].size
    return image

  def __wrap(self, xc) :
    while xc < 0: xc += self.ncells
    while xc >= self.ncells: xc -= self.ncells
    return xc

  def __build(self, field):
    sml = field['sml']
    pos = field['locations']

    boxsize = field.boxsize
    ncells = int(boxsize / max(sml))
    self.boxsize = boxsize
    self.cellsize = boxsize / ncells
    self.ncells = ncells

    hoc = [[] for i in range(ncells * ncells)]

    print "building hoc stage 1"

    ccode = r"""
  #line 41 "test.py"
  double cellsize = (double) boxsize / ncells;

  for(int i = 0; i < Npos[0]; i++) {
    double x = POS2(i, 0);
    double y = POS2(i, 1);
    int xcc = x / cellsize;
    int ycc = y / cellsize;
    while(xcc < 0) xcc += ncells;
    while(xcc >= ncells) xcc -= ncells;
    while(ycc < 0) ycc += ncells;
    while(ycc >= ncells) ycc -= ncells;
    PyObject * val = PyLong_FromLong(i);
    PyObject * entry = PyList_GET_ITEM(py_hoc, xcc * ncells + ycc);
    PyList_Append(entry, val);
    Py_DECREF(val);
  }
    """
    inline(ccode, ['boxsize', 'ncells', 'pos', 'sml', 'hoc'], extra_compile_args=["-Wno-unused"])

    print "building hoc stage 2, linking neighbours"
    for xc in range(ncells):
      for yc in range(ncells):
        xcl = self.__wrap(xc - 1)
        xcr = self.__wrap(xc + 1)
        ycl = self.__wrap(yc - 1)
        ycr = self.__wrap(yc + 1)
        indices = [ i * ncells + j for i in [xcl, xc, xcr] for j in [ycl, yc, ycr]]

        indices = unique(indices)
        result = []
        l = 0
        for i in indices : 
          l += len(hoc[i])
        a = zeros((l), dtype='u4')
        l = 0
        for i in indices :
          a[l:l+len(hoc[i])] = hoc[i]
          l += len(hoc[i])
        self.hoc[xc * ncells + yc] = a
    del hoc
