#! python
from gaepsi.field import Field
from gaepsi.snapshot import Snapshot
from gaepsi.cosmology import default as cosmology
from pylab import *

from sys import argv

boxsize = 6.6 * cosmology.h
N = 64
snapshot = Snapshot(argv[2], argv[1], create=True)

field = Field(numpoints = N**3, components={
     'vel':('f4', 3),
     'mass':'f4',
     'rho':'f4',
     'sml':'f4',
     'xHI':'f4',
     'ye':'f4',
     'ie':'f4'})

field.boxsize = boxsize
field.cosmology = cosmology

x = linspace(0, boxsize, N, endpoint=False)
x.shape = N, 1, 1

field['locations'][:,0] = tile(x, (N**2,1,1)).flat
field['locations'][:,1] = tile(x, (N,N,1)).flat
field['locations'][:,2] = tile(x, (1,1,N**2)).flat

#1e-3 proton per cm^3 comovming
density = 1e-3 * 1e6 * cosmology.units.PROTONMASS * cosmology.units.LENGTH ** 3
mass = density * boxsize ** 3
field['rho'][:] = density
field['mass'][:] = mass / N**3
field['xHI'][:] = 1.0
field['ye'][:] = 0.0
field['ie'][:] = 1e4 / cosmology.units.TEMPERATURE
field.smooth()

print 'critcial density', cosmology.units.CRITICAL_DENSITY
print 'cosmology', cosmology
print 'current density', density
print 'sml = ', field['sml'].mean(), 'separation = ', boxsize / (N * 1.0)
print 'proton count', mass / cosmology.units.PROTONMASS

field.dump_snapshots([snapshot], ptype=0)

snapshot.save_all()
