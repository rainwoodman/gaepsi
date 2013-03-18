from gaepsi.snapshot import Snapshot
from gaepsi.field import Field
from gaepsi.readers import Reader
from gaepsi.tools import packarray
from gaepsi.store import getcomponent
import sharedmem
import numpy

def profile(field, component, center, rmin, rmax, weights=None, logscale=True, nbins=100, density=True, integrated=True):
  """ returns centers, profile, and with of the bins, 
      if density == True, divide by volume 
      if integrated == True, use the sum of full enclose volume. otherwise use the shell"""
  locations = field['locations']
  component, weights = getcomponent(None, field, component, weights)
  if logscale:
    rmin = numpy.log10(rmin)
    rmax = numpy.log10(rmax)
  bins = numpy.linspace(rmin, rmax, nbins + 1, endpoint=True)
  if integrated:
    centers = bins[1:]
  else:
    centers = (bins[:-1] + bins[1:]) * 0.5
  if logscale:
    centers = 10 ** centers
    bins = 10 ** bins


  profil = numpy.zeros(nbins + 2, dtype='f8')
  weight = numpy.zeros(nbins + 2, dtype='f8') # for the profile

   
  with sharedmem.Pool(use_threads=True) as pool:
    def work(locations, component, weights):
      r = ((locations - center) ** 2).sum(axis=-1) ** 0.5
      dig = numpy.digitize(r, bins)
      if weights is not None:
        p = numpy.bincount(dig, weights=component * weights, minlength=nbins+2)
        w = numpy.bincount(dig, weights=weights, minlength=nbins+2)
        return p, w
      else:
        p = numpy.bincount(dig, weights=component, minlength=nbins+2)
        return p
    def reduce(res):
      if weights is not None:
        p, w = res
        profil[:] += p
        weight[:] += w
      else:
        profil[:] += res
    pool.starmap(work, pool.zipsplit((locations, component, weights)), callback=reduce)
  if integrated:
    profil = profil.cumsum()
    weight = weight.cumsum()

  if weights is not None:
    profil /= weight

  if density:
    if integrated:
      profil[1:-1] /= 4 / 3.0 * 3.1416 * bins[1:] ** 3
    else:
      profil[1:-1] /= numpy.diff(4 / 3.0 * 3.1416 * bins ** 3)

  return centers, profil[1:-1], numpy.diff(bins)[:-1]
  
class HaloCatalog(Field):
  def __init__(self, tabfilename, format, count=10):
    """ 
       tabfilename is like groups_019/group_tab_019.%d.
    """
    i = 0
    # decide number of files to open
    nread = 0
    tabs = []
    while nread < count:
      g = Snapshot(tabfilename % i, format + '.GroupTab')
      nread += g.C['N'][0] 
      i = i + 1
      tabs.append(g)
    Field.__init__(self, numpoints=count, components={'offset':'i8', 'length':'i8', 'massbytype':('f8', 6), 'mass':'f8'})
    self.take_snapshots(tabs, ptype=0)
    del tabs
    # fix the offset which may overflow for large halos
    self['offset'][1:] = self['length'].cumsum()[:-1]

    nread = 0
    nshallread = self['length'].sum()
    ids = []
    reader = Reader(format)
    i = 0
    while nread < nshallread:
      more = numpy.memmap(tabfilename.replace('tab', 'ids') % i, dtype=reader['id'].dtype, mode='r', offset=28)
      ids.append(more)
      nread += len(more)
      i = i + 1
    self.ids = packarray(numpy.concatenate(ids), self['length'])

  def mask(self, parids, groupid):
    ids = self.ids[groupid]
    ids.sort()
    return parids == ids[ids.searchsorted(parids).clip(0, len(ids) - 1)]

  def select(self, field, groupid):
    return field[self.mask(field['id'], groupid)]

class BHDetail:
  def __init__(self, filename, mergerfile=None, numfields=None):
    """ to combine bunch of bhdetails file into one file, run
     cat blackhole_details_* | awk '/^BH=/ {if (NF==14) print substr($0, 4)}' |sort -gk 2 > bhdetail.txt 
     cat blackhole_details_* |grep swallows | awk '{print substr($2, 6, length($2)-6), substr($3,4), $5}' > bhdetail-merger.txt
      everything will be in internal units
    """
    dtlist = [
     ('id', 'u8'),
     ('time', 'f8'),
     ('mass', 'f8'),
     ('mdot', 'f8'),
     ('rho', 'f8'),
     ('cs', 'f8'),
     ('vel', 'f8'),
     ('posx', 'f8'),
     ('posy', 'f8'),
     ('posz', 'f8'),
     ('vx', 'f8'), 
     ('vy', 'f8'),
     ('vz', 'f8'),
     ('sml', 'f8'),
     ('surrounding', 'f8'),
     ('dt', 'f8'),
     ('mainid', 'u8'),
     ('z', 'f8'),
    ]
    if numfields is None:
      numfields = len(file(filename).readline().split())
    rawdt = numpy.dtype(dtlist[:numfields])
    raw = numpy.loadtxt(filename, dtype=rawdt)
    data = numpy.empty(len(raw), dtlist)
    data[...] = raw
    data['mainid'] = data['id']
    data['z'] = 1 / data['time'] - 1
    if mergerfile is not None:
      merger = numpy.loadtxt(mergerfile, dtype=[('time_q', 'f8'), 
           ('after', 'u8'), ('swallowed', 'u8')], ndmin=1)
      while True:
        changed = False
        for time_q, after, swallowed in merger:
          changed = changed or (data['mainid'] == swallowed).any()
          data['mainid'][[data['mainid'] == swallowed]] = after
        if not changed: break
  
    data.sort(order=('mainid', 'time'))
    self.data = data

  def mainid(self, id):
    return self['mainid'][self['id'] == id][0]

  def mostmassive(self, id):
    """ construct a data series for the most massive bh of id at any time"""
    mask = self.data['mainid'] == self.mainid(id)
    data = self.data[mask]
    data.sort(order='time')
    while True:
      mass = data['mass']
      mask = mass[1:] > mass[:-1]
      if mask.all(): break
      data = data[1:][mask]
    return data

  def prog(self, id):
    """ returns a list of all progenitors """
    mask = self.data['mainid'] == self.mainid(id)
    ids = numpy.unique(self.data['id'][mask])
    return ids

  def __getitem__(self, index):
    """ either by id (not mainid) or by the component"""
    if isinstance(index, basestring):
      return self.data[index]
    if hasattr(index, '__iter__'):
      return [self[i] for i in index]
    else:
      return self.data[self.data['id'] == id]


  def plot(self, id, property, xdata=None, *args, **kwargs):
    """plot a series of blackholes, deprecated. use matplotlib
       and indexing directly
    """
    if hasattr(id, '__iter__'):
      return sum([ self.plot(i, property, xdata=xdata, *args, **kwargs) for i in id ], [])
    ax = kwargs.pop('ax', None)
    mask = (self.data['id'] == id)
  #plot(1./raw['time_q'][mask] - 1, numpy.convolve(raw[property][mask], numpy.ones(10) / 10, mode='same'), label='%d' % id, *args, **kwargs)
    if isinstance(property, basestring):
      property = self.data[property]
    if isinstance(xdata, basestring):
      xdata = self.data[xdata]
    if ax == None: 
      from matplotlib import pyplot
      ax = pyplot.gca()
    return ax.plot(xdata[mask], property[mask], *args, **kwargs)
