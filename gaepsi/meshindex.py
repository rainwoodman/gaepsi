import numpy
from gaepsi.readers import F77File
from gaepsi.tools import packarray

headerdtype = numpy.dtype(
    [ 
      ('version', 'i4'),
      ('Nd', 'i4'),
    ])

class MeshIndex:
    def __init__(self, N, boxsize, Nd=3):
        self.Nd = Nd
        self.boxsize = numpy.empty(Nd, dtype='f8')
        self.boxsize[:] = boxsize
        self.N = numpy.empty(Nd, dtype='i4')
        self.N[:] = N
        self.dict = {}
        pass

    def set(self, label, pos):
        assert pos.shape[1] == self.Nd
        cell = numpy.int64(pos / (self.boxsize / self.N)[None, :])
        index = numpy.ravel_multi_index(cell.T, self.N)
        return label, numpy.unique(index)

    def __getitem__(self, index):
        lin = numpy.ravel_multi_index(index, self.N)
        return self.data[index]

    def tofile(self, filename):
        f = F77File(filename, 'w')
        header = numpy.empty((), headerdtype)
        header['Nd'] = self.Nd
        header['version'] = 0
        f.write_record(header)
        f.write_record(self.N)
        f.write_record(self.boxsize)
        sizes = self.data.end - self.data.start
        size_out = numpy.int32(sizes.reshape(self.N[0], -1))
        start = self.data.start.reshape(self.N[0], -1)
        end = self.data.end.reshape(self.N[0], -1)
        for i in range(self.N[0]):
            f.write_record(size_out[i, ...])
        for i in range(self.N[0]):
            f.write_record(
                    self.data.A[start[i, 0]:
                        end[i, -1]])
        
    @classmethod
    def fromfile(cls, filename):
        f = F77File(filename, 'r')
        header = f.read_record(headerdtype, 1)[0]
        N = f.read_record('i4', header['Nd'])
        boxsize = f.read_record('f8', header['Nd'])
        self = cls(N=N, boxsize=boxsize, Nd=header['Nd'])

        size = numpy.empty(0, dtype='i4')
        data = numpy.empty(0, dtype='i4')
        for i in range(self.N[0]):
            size_in = f.read_record('i4', numpy.prod(self.N[1:]))
            size = numpy.append(size, size_in)

        size = size.reshape(self.N[0], -1)
        for i in range(self.N[0]):
            data_in = f.read_record('i4', size[i].sum(dtype='i8'))
            data = numpy.append(data, data_in)

        self.data = packarray(data, size.reshape(-1))
        return self

    def cut(self, origin, size):
        start = numpy.int32(origin / (self.boxsize / self.N))
        end = numpy.int32(numpy.ceil((origin + size) \
                / (self.boxsize / self.N)))
        indices = start[:, None] + numpy.indices(end - start).reshape(self.Nd,
                -1)
        ind = numpy.ravel_multi_index(indices, self.N)
        return numpy.unique(numpy.concatenate(self.data[ind]))
        
    def compile(self, d):
        """ takes a dict of (label, cellids) as input
            the list can be build concatenating
            the output of add(label, pos)
        """
        labels = numpy.array(list(d.keys()), dtype='int32')
        labels.sort()
        join = numpy.concatenate([d[label] for label in labels])
        sizes = numpy.bincount(join, minlength=numpy.prod(self.N))

        data = numpy.repeat(labels, 
                [d[label].size for label in labels])
        arg = join.argsort()
        self.data = packarray(data[arg], sizes)

