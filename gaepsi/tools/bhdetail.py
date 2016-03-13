import numpy
from gaepsi.tools import packarray
import os.path
import glob
import sharedmem

def uniqueclustered(data):
    """ find unique elements from clustered data.
      clustered data has identical elements arranged next to each other."""
    diff = (data[1:] != data[:-1]).nonzero()[0]
    start, end = numpy.empty((2, len(diff) + 1), numpy.intp)
    start[0] = 0
    start[1:] = diff + 1
    end[:-1] = diff + 1
    end[-1] = len(data)
    unique = data[start]
    return unique, start, end

def bhfilenameiter(filenames):
    i = 0
    while True:
        fn = filenames % i
        if not os.path.exists(fn): 
            if i == 0:
                raise IOError("file %s not found" % fn)
            break
        yield fn 
        i = i + 1

class BHDetail2:
    def __init__(self, path=None, every=1, raw=False):
        """ 
            path is the dirname of the path to the blackhole_details_?.txt files
            everything will be in internal units

            .trees is sorted by final blackhole mass [0] the most massive
            .trees.byid is a dictionary accessing the trees by id.
            .blackholes is sorted by final bh mass [0] the most massive
            .blackholes.byid is a dictionary accessing the blackholes by id.
        """
        if not raw and os.path.exists(os.path.join(path, "blackhole_details_%d.txt" % 0)):
            data, merger = self._readtxt(path, every)
        else:
            data, merger = self._readraw(path, every)
        self.data = data
        self.merger = merger
        self._fillmain()
        self._fillparent()
        self.data.sort(order=['mainid', 'id', 'time']) 

        # data is already clustered by mainid and id
        treeids, start, end = uniqueclustered(self.data['mainid'])
        trees = packarray(self.data, start=start, end=end)
        arg = numpy.argsort([tree['mass'].max() for tree in trees])[::-1]
        self.trees = packarray(self.data, start=start[arg], end=end[arg])
        self.trees.byid = dict(zip(treeids, trees))
        bhids, start, end = uniqueclustered(self.data['id'])
        blackholes = packarray(self.data, start=start, end=end) 
        arg = numpy.argsort([blackhole['mass'].max() for blackhole in
            blackholes])[::-1]
        self.blackholes = packarray(self.data, start=start[arg], end=end[arg])
        self.blackholes.byid = dict(zip(bhids, blackholes))
    #        self._fillmergermass()
        self.merger2 = self.merger.copy()
        self.merger2.sort(order=['after', 'time'])

        if(len(merger) > 0) :
            t = merger['time']
            arg = t.argsort()
            t = t[arg]
            after = merger['after'][arg]
            swallowed = merger['swallowed'][arg]
            ind = t.searchsorted(self.data['time'])
            bad = (t.take(ind, mode='clip') == self.data['time'])
            bad &= after.take(ind, mode='clip') == self.data['id']
            bad &= swallowed.take(ind, mode='clip') == self.data['id']
            self.data['mass'][bad] = numpy.nan

    def _readraw(self, path, every):
        dtype = numpy.dtype([
            ('type', 'i4'),
            ('', 'i4'),
            ('time', 'f8'),
            ('id', 'u8'),
            ('pos', ('f8', 3)),
            ('mass', 'f4'),
            ('mdot', 'f4'),
            ('rho', 'f4'),
            ('cs', 'f4'),
            ('bhvel', 'f4'),
            ('gasvel', ('f4', 3)),
            ('hsml', 'f4'),
            ('', 'f4')
            ])
        dtype1 = numpy.dtype([
            ('type', 'i4'),
            ('', 'i4'),
            ('time', 'f8'),
            ('id', 'u8'),
            ('pos', ('f8', 3)),
            ('mass', 'f4'),
            ('mdot', 'f4'),
            ('rho', 'f4'),
            ('cs', 'f4'),
            ('bhvel', 'f4'),
            ('velx', ('f4', 3)),
            ('hsml', 'f4'),
            ('z', 'f4'),
            ('mainid', 'u8'),
            ('parentid', 'u8')
            ])
        dtype2 = numpy.dtype([
            ('type', 'i4'),
            ('', 'i4'),
            ('time', 'f8'),
            ('after', 'u8'),
            ('swallowed', 'u8'),
            ('mbefore', 'f4'),
            ('mswallowed', 'f4'),
            ('bhvel', 'f4'),
            ('cs', 'f4'),
            ('padding', ('u1',  dtype.itemsize - 48))
            ])
        assert dtype2.itemsize == dtype.itemsize
        def work(filename):
            raw = numpy.fromfile(filename, dtype=dtype)
            data = numpy.empty((raw['type'] == 0).sum(), dtype=dtype1)
            data[:] = raw[raw['type'] == 0]

            data['z'] = 1 / data['time'] - 1
            return data, raw[raw['type'] == 2].view(dtype=dtype2)

        data0 = [numpy.array([], dtype=dtype1)]
        merger0 = [numpy.array([], dtype=dtype2)]

        def reduce(raw, mergerlist):
            data0[0] = numpy.append(data0[0], raw[::every])
            merger0[0] = numpy.append(merger0[0], mergerlist)

        try:
            filenames = list(bhfilenameiter(os.path.join(path, "blackhole_details_%d.raw")))
        except IOError:
            filenames = list(glob.glob(os.path.join(path, "dumpdir-*/blackhole_details_*.raw")))

        with sharedmem.MapReduce() as pool:
            pool.map(work, filenames, reduce=reduce)

        data = data0[0]
        data['z'] = 1 / data['time'] - 1

        merger = merger0[0]
        merger.sort(order=['swallowed', 'time'])
        return data, merger
        
    def _readtxt(self, path, every):
        dtype = numpy.dtype([
         ('id', 'u8'), ('time', 'f8'), ('mass', 'f4'), ('mdot', 'f4'),
         ('rho', 'f4'), ('cs', 'f4'), ('bhvel', 'f4'), ('pos', ('f8', 3)),
         ('vel', ('f4', 3)), ('hsml', 'f4'), ('surrounding', 'f4'), ('dt', 'f4'),
         ('mainid', 'u8'), ('parentid', 'u8'), ('z', 'f4'), ])

        def work(filename):
            mergerlist = []
            with file(filename, 'r') as f:
                def iter():
                    i = 0
                    for line in f:
                        if line.startswith('BH='): 
                            i = i + 1
                            if i != every: continue
                            else: i = 0
                            yield line[3:]
                        elif line.startswith('ThisTask='): 
                            words = line.split()
                            if words[3].startswith('swallows'): 
                                mergerlist.append('%s %s %s %s %s' % \
                                    (words[1][5:-1], words[2][3:], words[4], words[5][1:],
                                        words[6][:-1]))
                raw = sharedmem.loadtxt2(iter(), dtype=dtype)
                return raw, mergerlist    

        data0 = [numpy.array([], dtype=dtype)]
        mergerlistfull = []

        def reduce(raw, mergerlist):
            data0[0] = numpy.append(data0[0], raw)
            mergerlistfull.extend(mergerlist)

        filenames = list(bhfilenameiter(os.path.join(path, "blackhole_details_%d.txt")))
        with sharedmem.MapReduce() as pool:
            pool.map(work, filenames, reduce=reduce)

        data = data0[0]
        data['z'] = 1 / data['time'] - 1


        raw = numpy.loadtxt(mergerlistfull, dtype=[('time', 'f8'), 
           ('after', 'u8'), ('swallowed', 'u8'), 
           ('mbefore', 'f8'), 
           ('mswallowed', 'f8')], ndmin=1)
        merger = numpy.empty(len(raw), dtype=
                [('time', 'f8'), ('after', 'u8'), ('swallowed', 'u8'),
                    ('mbefore', 'f8'), ('mswallowed', 'f8')])
        merger[:] = raw
        merger.sort(order=['swallowed', 'time'])
        return data, merger

    def _fillmergermass(self):
        for entry in self.merger:
            bh = self.blackholes[entry['after']]
            arg = bh['time'].searchsorted(entry['time'])
            if entry['time'] == bh['time'][arg]: arg = arg - 1
            entry['mbefore'] = bh['mass'][arg]

            bh = self.blackholes[entry['swallowed']]
            entry['mswallowed'] = bh['mass'][-1]

    def getmostmassive(self, tree):
        out = []
        tree = tree[~numpy.isnan(tree['mass'])]
        arg = tree['mass'].argmax()
        finalid = tree['id'][arg]
        starttime = tree['time'][arg]
        while True:
            bh = self.blackholes.byid[finalid]
            left = self.merger2['after'].searchsorted(finalid, side='left')
            right = self.merger2['after'].searchsorted(finalid, side='right')
            merger2 = self.merger2[left:right][::-1]
            keep = merger2['mbefore'] >= merger2['mswallowed']

            if keep.all():
                out.append(bh[(bh['time'] <= starttime)])
                break
            else:
                stopind = (~keep).nonzero()[0][0]
                stoptime = merger2['time'][stopind]
                out.append(bh[(bh['time'] <= starttime) & (bh['time'] > stoptime)])
                finalid = merger2['swallowed'][stopind]
                starttime = stoptime

        out = numpy.concatenate(out)
        out.sort(order='time')
        assert (out['time'][1:] >= out['time'][:-1]).all()
        return out

    def gettotal(self, tree):
        starttime = tree['time'].min()
        endtime = tree['time'].max()
        grid = numpy.linspace(starttime, endtime, 128)
        out = numpy.empty(len(grid), dtype=self.data.dtype)

        sumfields = ['mass', 'mdot']
        meanfields = ['rho', 'cs', 'vel', 'posx', 'posy', 'posz', 'vx', 'vy',
        'vz', 'sml']

        raise "Not implemented"
    def _fillmain(self):
        data = self.data
        data['mainid'] = data['id']
        if len(self.merger) == 0: return
        while True:
            ind = self.merger['swallowed'].searchsorted(data['mainid'])
            ind[ind >= len(self.merger)] = len(self.merger) - 1
            found = self.merger['swallowed'][ind] == data['mainid']
            if not found.any(): 
                break
            data['mainid'][found] = self.merger['after'][ind[found]]
    def _fillparent(self):
        data = self.data
        data['parentid'] = data['id']
        if len(self.merger) == 0: return
        ind = self.merger['swallowed'].searchsorted(data['parentid'])
        ind[ind >= len(self.merger)] = len(self.merger) - 1
        found = self.merger['swallowed'][ind] == data['parentid']
        data['parentid'][found] = self.merger['after'][ind[found]]
  
    @classmethod
    def load(kls, filename):
        pass
    def save(self, filename):
        numpy.save(filename, [self.data, self.merger])

