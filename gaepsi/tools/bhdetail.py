import numpy
from gaepsi.tools import packarray

def uniqueclustered(data):
    diff = (data[1:] != data[:-1]).nonzero()[0]
    start, end = numpy.empty((2, len(diff) + 1), numpy.intp)
    start[0] = 0
    start[1:] = diff + 1
    end[:-1] = diff + 1
    end[-1] = len(data)
    unique = data[start]
    return unique, start, end

class BHDetail2:
    def __init__(self, filename, mergerfile=None, numfields=None):
        """ to combine bunch of bhdetails file into one file, run
         cat blackhole_details_* | awk '/^BH=/ {if (NF==14) print substr($0, 4)}' |sort -gk 2 > bhdetail.txt 
         cat blackhole_details_* |grep swallows | awk '{print substr($2, 6, length($2)-6), substr($3,4), $5}' > bhdetail-merger.txt
          everything will be in internal units
        """
        dtlist = [
         ('id', 'u8'), ('time', 'f8'), ('mass', 'f8'), ('mdot', 'f8'),
         ('rho', 'f8'), ('cs', 'f8'), ('vel', 'f8'), ('posx', 'f8'),
         ('posy', 'f8'), ('posz', 'f8'), ('vx', 'f8'), ('vy', 'f8'),
         ('vz', 'f8'), ('sml', 'f8'), ('surrounding', 'f8'), ('dt', 'f8'),
         ('mainid', 'u8'), ('parentid', 'u8'), ('z', 'f8'), ]
    
        if filename[-4:] == '.npy':
            self.data, self.merger = numpy.load(filename)
    
        else:
            if numfields is None:
              numfields = len(file(filename).readline().split())
            if numfields != 0: 
                rawdt = numpy.dtype(dtlist[:numfields])
            else:
                rawdt = numpy.dtype(dtlist)
            raw = numpy.loadtxt(filename, dtype=rawdt)
            data = numpy.empty(len(raw), dtlist)
            data[...] = raw
            del raw
            data['z'] = 1 / data['time'] - 1
    
            self.data = data
            if mergerfile is not None:
                raw = numpy.loadtxt(mergerfile, dtype=[('time', 'f8'), 
                   ('after', 'u8'), ('swallowed', 'u8')], ndmin=1)
                merger = numpy.empty(len(raw), dtype=
                        [('time', 'f8'), ('after', 'u8'), ('swallowed', 'u8'),
                            ('mafter', 'f8'), ('mswallowed', 'f8')])
                merger[:] = raw
                merger.sort(order=['swallowed', 'time'])
                self.merger = merger
            self._fillmain()
            self._fillparent()
            self.data.sort(order=['mainid', 'id', 'time']) 

        # data is already clustered by mainid and id
        treeids, start, end = uniqueclustered(self.data['mainid'])
        trees = packarray(self.data, start=start, end=end)
        self.trees = dict(zip(treeids, trees))
        bhids, start, end = uniqueclustered(self.data['id'])
        blackholes = packarray(self.data, start=start, end=end) 
        self.blackholes = dict(zip(bhids, blackholes))
        self._fillmergermass()
        self.merger2 = self.merger.copy()
        self.merger2.sort(order=['after', 'time'])


    def _fillmergermass(self):
        for entry in self.merger:
            bh = self.blackholes[entry['after']]
            arg = bh['time'].searchsorted(entry['time'])
            if entry['time'] == bh['time'][arg]: arg = arg - 1
            entry['mafter'] = bh['mass'][arg]

            bh = self.blackholes[entry['swallowed']]
            entry['mswallowed'] = bh['mass'][-1]

    def getmostmassive(self, tree):
        out = []
        arg = tree['mass'].argmax()
        finalid = tree['id'][arg]
        starttime = tree['time'][arg]
        while True:
            bh = self.blackholes[finalid]
            left = self.merger2['after'].searchsorted(finalid, side='left')
            right = self.merger2['after'].searchsorted(finalid, side='right')
            merger2 = self.merger2[left:right][::-1]
            keep = merger2['mafter'] >= merger2['mswallowed']

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
  
    def save(self, filename):
        numpy.save(filename, [self.data, self.merger])

