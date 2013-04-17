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
  def __init__(self, tabfilename, format, count=10, **kwargs):
    """ 
       tabfilename is like groups_019/group_tab_019.%d.
    """
    i = 0
    # decide number of files to open
    nread = 0
    tabs = []
    while nread < count:
      g = Snapshot(tabfilename % i, format + '.GroupTab',
              **kwargs)
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
    reader = Reader(format, **kwargs)
    i = 0
    while nread < nshallread:
      more = numpy.memmap(tabfilename.replace('tab', 'ids')
              % i, dtype=g.C['idtype'], mode='r', offset=28)
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

    if filename[-4:] == '.npy':
        self.data = numpy.load(filename)
        return

    if numfields is None:
      numfields = len(file(filename).readline().split())
    rawdt = numpy.dtype(dtlist[:numfields])
    raw = numpy.loadtxt(filename, dtype=rawdt)
    data = numpy.empty(len(raw), dtlist)
    data[...] = raw
    del raw
    data['mainid'] = data['id']
    data['z'] = 1 / data['time'] - 1

    self.data = data

    if mergerfile is not None:
      merger = numpy.loadtxt(mergerfile, dtype=[('time_q', 'f8'), 
           ('after', 'u8'), ('swallowed', 'u8')], ndmin=1)
      while self.fixmainid(merger) > 0:
        pass

  def fixmainid(self, merger):
    arg = sharedmem.argsort(self.data['mainid'])
    self.data[...] = self.data[arg]

    left = self.data['mainid'].searchsorted(merger['swallowed'], side='left')
    right = self.data['mainid'].searchsorted(merger['swallowed'], side='right')
    mask = self.data['mainid'][left] == merger['swallowed']
    print 'fixing mainid', mask.sum(), 'remaining'
    left = left[mask]
    right = right[mask]
    for i, row in enumerate(merger[mask]):
      time, after, swallowed = row
      l = left[i]
      r = right[i]
      self.data['mainid'][l:r+1] = after
    return mask.sum()

  def save(self, filename):
    numpy.save(filename, self.data)
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


def cic(pos, Nmesh, boxsize, weights=1.0, dtype='f8'):
    """ CIC approximation from points to Nmesh,
        each point has a weight given by weights.
        This does not give density.
        pos is supposed to be row vectors. aka for 3d input
        pos.shape is (?, 3).

    """
    chunksize = 1024 * 16
    BoxSize = 1.0 * boxsize
    Ndim = pos.shape[-1]
    Np = pos.shape[0]
    dtype = numpy.dtype(dtype)
    mesh = numpy.zeros(shape=(Nmesh, ) * Ndim, 
            dtype=dtype, order='C')
    flat = mesh.reshape(-1)
    neighbours = ((numpy.arange(2 ** Ndim)[:, None] >> \
            numpy.arange(Ndim)[None, :]) & 1)
    for start in range(0, Np, chunksize):
        chunk = slice(start, start+chunksize)
        if numpy.isscalar(weights):
          wchunk = weights
        else:
          wchunk = weights[chunk]
        gridpos = pos[chunk] * (Nmesh / BoxSize)
        intpos = numpy.intp(gridpos)
        for i, neighbour in enumerate(neighbours):
            neighbour = neighbour[None, :]
            targetpos = intpos + neighbour
            targetindex = numpy.ravel_multi_index(
                    targetpos.T, mesh.shape, mode='wrap')
            kernel = (1.0 - numpy.abs(gridpos - targetpos)).prod(axis=-1)
            add = wchunk * kernel
            u, label = numpy.unique(targetindex, return_inverse=True)
            flat[u] += numpy.bincount(label, add)
    return mesh

def collapse(field, ticks, axis, logscale=False):
    """ collapse axis of a field,
        tics are the coordinates of the axes.
        tics is of length field.shape
        and len(tics[i]) == field.shape[i]
        
        axis is a list of axis to collapse.
        the collapsed axis will be represented by one
        axis added to the end of the axes of the return
        value.
        returns

           tics, newfield
        where tics are the coordinates of the new field.
    """
    Ndim = len(field.shape)
    for i in range(Ndim):
      assert len(ticks[i]) == field.shape[i]

    if axis is None or len(axis) == 0:
        return ticks, field

    axis = list(axis)

    preserve = []
    newticks = []
    dist = None
    for i in range(Ndim):
        if i in axis:
            if dist is None:
                dist = ticks[i] ** 2
            else:
                dist = dist[:, None] + ticks[i][None, :] ** 2
            dist.shape = -1
        else:
            preserve.append(i)
            newticks.append(ticks[i])
    dist **= 0.5
    dmin = dist[dist > 0].min()     
    dmax = dist.max()

    Nbins = numpy.max(numpy.array(field.shape)) + 1
    if not logscale:
        bins = numpy.linspace(dmin, dmax, Nbins, endpoint=True)
        center = 0.5 * (bins[1:] + bins[:-1])
    else:
        ldmin, ldmax = numpy.log10([dmin, dmax])
        bins = numpy.logspace(ldmin, ldmax, Nbins, endpoint=True)
        center = (bins[1:] * bins[:-1]) ** 0.5

    newticks.append(center)

    slabs = field.transpose(preserve + axis).reshape(-1, len(dist))

    dig = numpy.digitize(dist, bins)
    suminv = 1.0 / numpy.bincount(dig, weights=dist, minlength=bins.size+1)[1:-1]

    newfield = numpy.empty((slabs.shape[0], len(center)))

    for i, slab in enumerate(slabs):
        kpk = dist * slab
        kpksum = numpy.bincount(dig, weights=kpk, minlength=bins.size+1)[1:-1]
        newfield[i] = kpksum * suminv

    newfield.shape = [field.shape[i] for i in preserve] + [len(center)]

    return newticks, newfield

def powerfromdelta(delta, boxsize, logscale=False, collapse=None):
    """ delta is over density.
        this obviously does not correct for redshift evolution.
        returns a collapsed powerspectrum, 

        if collapse is None, collapse all dimensions.
        if collapse is not None, just collapse the selecte dimensions.
        if collapse is False, do not collapse and return the full n-dim
        Pk.

        returns k, P

        where k is a list of the bin centers.
           k[0] is the bins centers of the first uncollapsed dimension,
           k[-1] is the centers of the collapsed dimensions,
           when collapse is None, k is just the center of the collapsed
           dimensions.
        P is of shape (uncollapsed ... axis, len(k))
    
       The power spectrum is assumed to have the gadget convention,
       AKA, normalized to (2 * pi) ** -3 times sigma_8.
    """
    Dplus = 1.0 # no evolution correction
  
    N = numpy.prod(delta.shape, dtype='f8')
    Ndim = len(delta.shape)
    BoxSize = numpy.empty(Ndim, dtype='f8')
    BoxSize[:] = boxsize
  
    # each dim has a different K0
    K0 = 2 * numpy.pi / BoxSize
  
    delta_k = numpy.fft.rfftn(delta) / N
    intK = numpy.ogrid[[slice(0, n) for n in delta_k.shape]]
  
    half = numpy.array(delta_k.shape) // 2
    # last dim is already halved
    half[-1] = delta_k.shape[-1]
  
    if collapse is None:
        collapse = range(Ndim)
    if collapse == False:
        collapse = []
    collapse = list(collapse)
    original = []
    Kclps = 0
    Kret = []
    for i in range(Ndim):
        kx = (half[i] - numpy.abs(half[i] - intK[i])) * K0[i]
        if i in collapse:
            Kclps = Kclps + kx ** 2
        else:
            Kret.append(kx.ravel().copy())
            original.append(i)

    if len(collapse) == 0:
        return Kret, numpy.abs(delta_k) ** 2 * K0.prod() ** -1 * Dplus ** 2

    #this shall be calling collapse instead!
    # now we bin Kclps
    Kclps = Kclps ** 0.5
    kmax = Kclps.max() 
    kmin = Kclps[Kclps > 0].min()
    if not logscale:
        kbins = numpy.linspace(kmin, kmax, numpy.max(half), endpoint=True)
        kcenter = 0.5 * (kbins[1:] + kbins[:-1])
    else:
        lkmin, lkmax = numpy.log10([kmin, kmax])
        kbins = numpy.logspace(lkmin, lkmax, numpy.max(half), endpoint=True)
        kcenter = (kbins[1:] * kbins[:-1]) ** 0.5

    Kret.append(kcenter)

    slabs = delta_k.transpose(original + collapse).reshape([-1] + 
            [delta_k.shape[i] for i in collapse])
    Kclps = Kclps.transpose(original + collapse).reshape(
            [Kclps.shape[i] for i in collapse])

    dig = numpy.digitize(Kclps.ravel(), kbins)
    ksuminv = 1.0 / numpy.bincount(dig, weights=Kclps.ravel(), minlength=kbins.size+1)[1:-1]

    pk = numpy.empty((slabs.shape[0], len(kcenter)))

    for i, slab in enumerate(slabs):
        kpk = Kclps * (numpy.abs(slab) ** 2) # * K0 ** -3 * Dplus ** 2
        kpksum = numpy.bincount(dig, weights=kpk.ravel(), minlength=kbins.size+1)[1:-1]
        pk[i] = kpksum * ksuminv

    pk = pk.reshape([delta_k.shape[i] for i in original] + [pk.shape[-1]])

    if len(Kret) == 1: 
        Kret = Kret[0]

    return Kret, pk * K0.prod() ** -1 * Dplus ** 2

def corrfrompower(K, P, logscale=False, R=None):
    """calculate correlation function from power spectrum,
       P is 1d powerspectrum. if R is not None, estimate at
       those points .

       returns R, xi(R)

       internally this does a numerical integral with the trapz
       rule for the radial direction of the fourier transformation,
       with a gaussian damping kernel (learned from Xiaoying) 
       the nan points of P is skipped. (usually when K==0, P is nan)

       input power spectrum is assumed to have the gadget convention,
       AKA, normalized to (2 * pi) ** -3 times sigma_8.

       The integral can also be done on log K instead of K if logscale 
       is True.
    """
    mask = ~numpy.isnan(P) & (K > 0)
    K = K[mask]
    P = P[mask]
    P = P * (2 * numpy.pi) ** 3 # going from GADGET to xiao
    if R is None:
      R = 2 * numpy.pi / K
    if logscale:
      weight = K * numpy.exp(-K**2)
      diff = numpy.log(K)
    else:
      weight = numpy.exp(-K**2)
      diff = K
    XI = [4 * numpy.pi / r * \
        numpy.trapz(P * numpy.sin(K * r) * K * weight, diff) for r in R]
    XI = (2 * numpy.pi) ** -3 * numpy.array(XI)
  
    return R, XI
