from gaepsi.snapshot import Snapshot
from gaepsi.field import Field
from gaepsi.readers import Reader
from gaepsi.tools import packarray
from gaepsi.store import getcomponent
import sharedmem
import numpy
try:
    import pyfftw
    irfftn = pyfftw.interfaces.numpy_fft.irfftn
    rfftn = pyfftw.interfaces.numpy_fft.rfftn
    fftshift = pyfftw.interfaces.numpy_fft.fftshift
    fftfreq = pyfftw.interfaces.numpy_fft.fftfreq
except ImportError:
    irfftn = numpy.fft.irfftn
    rfftn = numpy.fft.rfftn
    fftshift = numpy.fft.fftshift
    fftfreq = numpy.fft.fftfreq

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

   
  with sharedmem.TPool() as pool:
    chunksize = 1024 * 1024
    def work(i):
      sl = slice(i, i + chunksize)
      r = ((locations[sl] - center) ** 2).sum(axis=-1) ** 0.5
      dig = numpy.digitize(r, bins)
      if weights is not None:
        p = numpy.bincount(dig, weights=component[sl] * weights[sl], minlength=nbins+2)
        w = numpy.bincount(dig, weights=weights[sl], minlength=nbins+2)
        return p, w
      else:
        p = numpy.bincount(dig, weights=component[sl], minlength=nbins+2)
        return p, None
    def reduce(p, w):
      if weights is not None:
        profil[:] += p
        weight[:] += w
      else:
        profil[:] += p
    pool.map(work, range(0, len(locations), chunksize), reduce=reduce)
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
    g = Snapshot(tabfilename % 0, format + '.GroupTab',
              **kwargs)
    if count < 0 or count > g.C['Ntot'][0]:
        count = g.C['Ntot'][0]
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
    #print 'will read', len(tabs), 'files'
    Field.__init__(self, numpoints=count, components={'offset':'i8',
        'length':'i8', 'massbytype':('f8', 6), 'mass':'f8', 'pos':('f8', 3),
        'vel':('f8', 3)})
    if len(tabs) > 0:
        self.take_snapshots(tabs, ptype=0)
        del tabs

        # fix the offset which may overflow for large halos
        self['offset'][1:] = self['length'].cumsum()[:-1]

        nread = 0
        nshallread = self['length'].sum()
        i = 0
        idslen = numpy.zeros(g.C['Nfiles'], dtype='i8')
        while nread < nshallread:
          idslen[i] = numpy.fromfile(tabfilename.replace('_tab_', '_ids_')
                  % i, dtype='i4', count=3)[2]
          nread += idslen[i]
          i = i + 1
        idsoffset = numpy.concatenate(([0], idslen.cumsum()))

        ids = sharedmem.empty(idslen.sum(), dtype=g.C['idtype'])

        #print 'reading', i, 'id files'

        with sharedmem.Pool() as pool:
          def work(i):
            more = numpy.memmap(tabfilename.replace('_tab_', '_ids_')
                  % i, dtype=g.C['idtype'], mode='r', offset=28)
            ids[idsoffset[i]:idsoffset[i] + idslen[i]] = more
          pool.map(work, range(i))
        self.ids = packarray(ids, self['length'])
        for i in range(self.numpoints):
          self.ids[i].sort()

  def mask(self, parids, groupid):
    ids = self.ids[groupid]
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
     ('parentid', 'u8'),
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
      self.fixparentmass(merger)
      while self.fixmainid(merger) > 0:
        pass
  def fixparentmass(self, merger):
    arg = sharedmem.argsort(self.data['id'])
    self.data[...] = self.data[arg]
    left = self.data['id'].searchsorted(merger['swallowed'], side='left')
    right = self.data['id'].searchsorted(merger['swallowed'], side='right')
    mask = self.data['id'][left] == merger['swallowed']
    left = left[mask]
    right = right[mask]
    print 'fix mass', mask.sum()
    for i, row in enumerate(merger[mask]):
      time, after, swallowed = row
      l = left[i]
      r = right[i]
      last = numpy.abs(self.data['time'][l:r] - time).argsort()[:1]
      #assert (self.data['time'][l:r] == time).any()
      #assert (self[after]['time'] == time).any()
      aftermask = self['id'] == after
      match = numpy.abs(self.data[aftermask]['time'] - time).argsort()[:1]

      m = numpy.max([self.data['mass'][l:r][last].max(), self.data[aftermask]['mass'][match].max()])
      self.data['mass'][l:r][last] = m # numpy.nan
      self.data['mass'][aftermask.nonzero()[0][match]] = m #numpy.nan
#      assert numpy.isnan(self.data['mass'][aftermask.nonzero()[0][match]]).all() # numpy.nan

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
      self.data['mainid'][l:r] = after
      #assert (self.data['time'][l:r] == time).any()
      #assert (self[after]['time'] == time).any()

    return mask.sum()

  def save(self, filename):
    numpy.save(filename, self.data)
  def mainid(self, id):
    return self['mainid'][self['id'] == id][0]

  def unique(self):
    return numpy.unique(self['mainid'])

  def finalmass(self, id):
      if numpy.isscalar(id):
          return numpy.nanmax(self.data['mass'][self.data['mainid'] ==
              self.mainid(id)])
      else:
          return [self.finalmass(i) for i in id]

  def mostmassive(self, id):
    """ construct a data series for the most massive bh of id at any time"""
    mask = self.data['mainid'] == self.mainid(id)
    mask &= ~numpy.isnan(self.data['mass'])
    data = self.data[mask]
    data.sort(order='time')
    output = []
    while True:
        id = data['id'][data['mass'].argmax()]
        sel = data[data['id'] == id]
        if len(sel) == 0: break
        output.append(sel)
        data = data[data['time'] < sel['time'].min()]
        if len(data) == 0: break
    return numpy.concatenate(output)

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
      return self.data[self.data['id'] == index]

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
        gridpos = numpy.remainder(pos[chunk], BoxSize) * (Nmesh / BoxSize)
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

def collapse(field, ticks=None, axis=[], logscale=False):
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
    if ticks is None:
        ticks = [1.0 * numpy.arange(i) for i in field.shape]
    for i in range(Ndim):
      assert len(ticks[i]) == field.shape[i]

    if axis is None:
        axis = list(range(Ndim))

    if len(axis) == 0:
        return ticks, field

    axis = list(axis)

    preserve = []
    newticks = []
    dist = None
    binsize = 0
    for i in range(Ndim):
        if i in axis:
            if dist is None:
                dist = ticks[i] ** 2
                binsize = ticks[i].ptp() / field.shape[i]
            else:
                dist = dist[:, None] + ticks[i][None, :] ** 2
                binsize = max(binsize, ticks[i].ptp() / field.shape[i])
            dist.shape = -1
        else:
            preserve.append(i)
            newticks.append(ticks[i])

    dist **= 0.5
    dmin = 0
    dmax = dist.max()
    Nbins = dmax / binsize
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
#    suminv = 1.0 / numpy.bincount(dig, weights=dist, minlength=bins.size+1)[1:-1]
    suminv = 1.0 / numpy.bincount(dig, minlength=bins.size+1)[1:-1]
    newfield = numpy.empty((slabs.shape[0], len(center)))
    for i, slab in enumerate(slabs):
        kpk = slab #* dist
        kpksum = numpy.bincount(dig, weights=kpk, minlength=bins.size+1)[1:-1]
        newfield[i] = kpksum * suminv

    newfield.shape = [field.shape[i] for i in preserve] + [len(center)]

    return newticks, newfield


def corrfromdelta(delta, boxsize, deconvolve=False, collapse_axes=None):
    K, P = powerfromdelta(delta, boxsize, deconvolve=deconvolve, collapse_axes=[])
    return corrfrompower(P, boxsize, collapse_axes)

def corrfrompower(P, boxsize, collapse_axes=None):

    if collapse_axes is None:
        collapse_axes = numpy.arange(len(P.shape))

    XI = irfftn(P)
    N = numpy.prod(XI.shape, dtype='f8')
    Ndim = len(XI.shape)
    BoxSize = numpy.empty(Ndim, dtype='f8')
    BoxSize[:] = boxsize
    print 'BoxSize', BoxSize

    if len(collapse_axes) != 0:
        XI = fftshift(XI)
    del P
    # in numpy, irfft * fft == 1, thus irfftn is actually DFT / N
    # the factor 2pi cancels with the convention in powerfromdelta
    # the remaining BoxSize is from Dk
    XI *= (N / numpy.product(BoxSize))
    X = []
    for i in range(len(XI.shape)):
        x = fftfreq(XI.shape[i])
        if len(collapse_axes) != 0:
            x = fftshift(x)
        x *= BoxSize[i]
        X.append(x)

    return collapse(XI, X, collapse_axes, logscale=False)

def cicdeconvolve(delta, boxsize=1.0):
    N = numpy.prod(delta.shape, dtype='f8')
    Ndim = len(delta.shape)
    BoxSize = numpy.empty(Ndim, dtype='f8')
    BoxSize[:] = boxsize
  
    # each dim has a different K0
    # (Dx)^3 = N / prod(BoxSize)
    # extra 2 * pi is from K0!
    K0 = 2 * numpy.pi / BoxSize
  
    delta_k = rfftn(delta) / N
  
    full = numpy.array(delta_k.shape)
    half = numpy.array(delta_k.shape) // 2
    # last dim is already halved
    half[-1] = delta_k.shape[-1] - 1
    full[-1] = half[-1] * 2
    Kret = []

    for i in range(Ndim):
        kx = fftfreq(full[i]) * full[i]
        kx = kx[:delta_k.shape[i]]
        Kret.append(kx * K0[i])

    for dim, ki in enumerate(Kret):
        shape = numpy.ones(len(Kret), dtype='i4')
        shape[dim] = len(ki)
        ki = ki.reshape(shape)
        kernel = numpy.sinc(ki * (0.5 * BoxSize[i] / full[dim]) / numpy.pi) ** -2
        delta_k *= kernel
    
    return irfftn(delta_k)

def powerfromdelta(delta, boxsize, logscale=False, collapse_axes=None,
        deconvolve=False, cheat=None):
    """ delta is over density.
        this obviously does not correct for redshift evolution.
        returns a collapsed powerspectrum, 

        if collapse_axes is None, collapse all dimensions.
        if collapse_axes is not None, just collapse the selecte dimensions.
        if collapse_axes is [], do not collapse and return the full n-dim
        Pk. [in fft freq orderings, 0 freq is at 0, not at center ]

        returns k, P

        where k is a list of the bin centers.
           k[0] is the bins centers of the first uncollapsed dimension,
           k[-1] is the centers of the collapsed dimensions,
           when collapse is None, k is just the center of the collapsed
           dimensions.
        P is of shape (uncollapsed ... axis, len(k))
    
       The power spectrum is assumed to have the gadget convention,
       AKA, normalized to (2 * pi) ** -3 times sigma_8.

       if deconvolve is True, deconvolve the field in K space with
       the CIC kernel.
    """
    Dplus = 1.0 # no evolution correction
  
    N = numpy.prod(delta.shape, dtype='f8')
    Ndim = len(delta.shape)
    BoxSize = numpy.empty(Ndim, dtype='f8')
    BoxSize[:] = boxsize
  
    # each dim has a different K0
    # (Dx)^3 = N / prod(BoxSize)
    # extra 2 * pi is from K0!
    K0 = 2 * numpy.pi / BoxSize
  
    if cheat is None:
        delta_k = rfftn(delta) / N
    else:
        delta_k = cheat

    if collapse_axes is None:
        collapse_axes = numpy.arange(len(delta_k.shape))
  
    full = numpy.array(delta_k.shape)
    half = numpy.array(delta_k.shape) // 2
    # last dim is already halved
    half[-1] = delta_k.shape[-1] - 1
    full[-1] = half[-1] * 2
    Kret = []

    for i in range(Ndim):
        kx = fftfreq(full[i]) * full[i]
        if i != Ndim - 1:
            if len(collapse_axes) != 0:
                kx = fftshift(kx)
        else:
            kx = kx[:delta_k.shape[i]]
        Kret.append(kx * K0[i])

    if len(collapse_axes) != 0:
        delta_k = fftshift(delta_k, axes=numpy.arange(len(delta.shape) - 1))

    if deconvolve:
        for dim, ki in enumerate(Kret):
            shape = numpy.ones(len(Kret), dtype='i4')
            shape[dim] = len(ki)
            ki = ki.reshape(shape)
            kernel = numpy.sinc(ki / K0[dim] / full[dim]) ** -2
            delta_k *= kernel

    P = numpy.abs(delta_k) ** 2 * K0.prod() ** -1 * Dplus ** 2
    if len(collapse_axes) == 0:
        return Kret, P
    else:
        return collapse(P, Kret, collapse_axes, logscale)


def corrfrompower_old(K, P, logscale=False, R=None):
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
    Pfunc = interp1d(K, P, kind=5)
    K = numpy.linspace(K.min(), K.max(), 1000000)
    P = Pfunc(K)

    if R is None:
      R = 2 * numpy.pi / K
    if logscale:
      weight = K #* numpy.exp(-K**2)
      diff = numpy.log(K)
    else:
      weight = 1 # numpy.exp(-K**2)
      diff = K
    XI = [4 * numpy.pi / r * \
        numpy.trapz(P * numpy.sin(K * r) * K * weight, diff) for r in R]
    XI = (2 * numpy.pi) ** -3 * numpy.array(XI)
  
    return R, XI

from scipy.interpolate import InterpolatedUnivariateSpline
class interp1d(InterpolatedUnivariateSpline):
  """ this replaces the scipy interp1d which do not always
      pass through the points
      note that kind has to be an integer as it is actually
      a UnivariateSpline.
  """
  def __init__(self, x, y, kind, bounds_error=False, fill_value=numpy.nan, copy=True):
    if copy:
      self.x = x.copy()
      self.y = y.copy()
    else:
      self.x = x
      self.y = y
    InterpolatedUnivariateSpline.__init__(self, self.x, self.y, k=kind)
    self.xmin = self.x[0]
    self.xmax = self.x[-1]
    self.fill_value = fill_value
    self.bounds_error = bounds_error
  def __call__(self, x):
    x = numpy.asarray(x)
    shape = x.shape
    x = x.ravel()
    bad = (x > self.xmax) | (x < self.xmin)
    if self.bounds_error and numpy.any(bad):
      raise ValueError("some values are out of bounds")
    y = InterpolatedUnivariateSpline.__call__(self, x.ravel())
    y[bad] = self.fill_value
    return y.reshape(shape)

def regulate(x, y, N, min=None, max=None, skipnan=False):
    """
        regulate data into N bins along x direction.
        the mean is saved.
        returns newx, meany
    """
    if skipnan:
        bad = numpy.isnan(x) | numpy.isnan(y)
        x = x[~bad]
        y = y[~bad]
    if min is None:
        min = x.min()
    if max is None:
        max = x.max()
    bins = numpy.linspace(min, max, N + 1, endpoint=True)
    yw = splat(x, y, bins)
    w = splat(x, 1, bins)
    return .5 * (bins[1:] + bins[:-1]), yw[1:-1] / w[1:-1]

def splat(t, value, bins):
    """put value into bins according to t
       the points are assumed to be describing a continuum field,
       if two points have the same position, they are merged into one point

       for points crossing the edge part is added to the left bin
       and part is added to the right bin.
       the sum is conserved.
    """
    if len(t) == 0:
        return numpy.zeros(len(bins) + 1)
    t = numpy.float64(t)
    t, label = numpy.unique(t, return_inverse=True)
    if numpy.isscalar(value):
        value = numpy.bincount(label) * value
    else:
        value = numpy.bincount(label, weights=value)
    edge = numpy.concatenate(([t[0]], (t[1:] + t[:-1]) * 0.5, [t[-1]]))
    dig = numpy.digitize(edge, bins)
    #use the right edge as the reference
    ref = bins[dig[1:] - 1]
    norm = (edge[1:] - edge[:-1])
    assert ((edge[1:] - edge[:-1]) > 0).all()
    norm = 1 / norm
    weightleft = -(edge[:-1] - ref) * norm
    weightright = (edge[1:] - ref) * norm
    # when dig < 1 or dig >= len(bins), t are out of bounds and does not
    # contribute.
    l = numpy.bincount(dig[:-1], value * weightleft, minlength=len(bins)+1)
    r = numpy.bincount(dig[1:], value * weightright, minlength=len(bins)+1)
    return l + r

def simplesmooth(mass, meandensity):
    """simply assign smoothing length for mass,
       acoording to the overdensity.

       Hsml = (3 / (4 pi) (32 * mass / density)) ** (1/3)
    """
    return 3.0 / (4.0 * 3.14) * (32 * mass / meandensity) ** 0.333
