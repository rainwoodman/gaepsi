#! /usr/bin/env python
import numpy
from gaepsi.store import Snapshot, Store, Field, create_cosmology
import argparse

from gaepsi.field import Field
from gaepsi.snapshot import Snapshot
from gaepsi.cosmology import Cosmology, WMAP7
from sys import argv

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
     '-L', '--boxsize', nargs='+', default=None,
     required=True, dest='boxsize', type=float)
  parser.add_argument(
     '--origin', nargs='+', default=None,
     required=True, dest='origin', type=float)
  parser.add_argument(
     '-N', '--nperside', default=None, 
      required=True, dest='nperside', type=int)
  parser.add_argument(
     '-f', '--format', default=None, 
     required=True, dest='format')
  parser.add_argument('-o', '--output', required=True, dest='output')
  parser.add_argument('template')
  group = parser.add_argument_group()
  group.add_argument('-T', '--temperature', required=False,
               dest='temperature', default=1e4, type=float)
  d = group.add_mutually_exclusive_group()
  d.add_argument('-D', '--overdensity', required=False,
               dest='overdensity', default=1.0, type=float)
  d.add_argument('-H', '--hydrogen', required=False,
               dest='hydrogen', default=None, type=float,
               metavar='atom per comoving cm*3')
  d.add_argument('--gasmass', required=False,
               dest='gasmass', default=None, type=float,
               metavar='gas particle mass')
  group.add_argument('--xHI', default=1.0, type=float, dest='xHI')
  group.add_argument('--ye', default=0.0, type=float, dest='ye')
  group = parser.add_argument_group()
  group.add_argument('--bhmass', default=1e-5, type=float, dest='bhmass')
  group.add_argument('--bhmdot', default=0, type=float, dest='bhmdot')

  args = parser.parse_args()

  if args.template is not None:
    template = Snapshot(args.template, args.format)
    C = template.C
  else:
    reader = Reader(args.format)
    C = reader.create_constants()

  if args.origin is not None:
    origin = numpy.empty(3)
    origin[...] = args.origin
  if args.boxsize is not None:
    boxsize = numpy.empty(3)
    boxsize[...] = args.boxsize
  if args.nperside is not None:
    nperside = numpy.empty(3, dtype='i8')
    nperside[...] = args.nperside

  snapshot = Snapshot(args.output, args.format, create=True)

  cosmology = create_cosmology(C)

  if args.hydrogen is not None:
    density = args.hydrogen * 1e6 * cosmology.units.PROTONMASS * cosmology.units.LENGTH ** 3
    hydrogen = args.hydrogen
    mass = density * boxsize.prod() / nperside.prod()
  elif args.gasmass is not None:
    mass = args.gasmass
    density = mass * nperside.prod() / boxsize.prod()
    hydrogen = density / 1e6 / ( cosmology.units.PROTONMASS * cosmology.units.LENGTH ** 3)
  else:
    density = args.overdensity * cosmology.units.CRITICAL_DENSITY * C['OmegaB']
    hydrogen = density / 1e6 / ( cosmology.units.PROTONMASS * cosmology.units.LENGTH ** 3)
    mass = density * boxsize.prod() / args.nperside.prod()


  print 'critcial density', cosmology.units.CRITICAL_DENSITY
  print 'cosmology', cosmology
  print 'current density', density, 
  print 'overdensity', density / (cosmology.units.CRITICAL_DENSITY * C['OmegaB'])
  print 'hydrogen per cm3', hydrogen
  print 'gas particle mass', mass

  field = Field(numpoints = nperside.prod(), components= {
     'vel':('f4', 3),
     'id':'u8',
     'mass':'f4',
     'rho':'f4',
     'sml':'f4',
     'xHI':'f4',
     'ye':'f4',
     'ie':'f4'})

  x = numpy.indices(nperside).reshape(3, -1).T


  field['locations'][:] = x * (boxsize / nperside)[None, :] + origin[None, :]

  field['rho'][:] = density
  field['id'][:] = range(0, nperside.prod())
  field['mass'][:] = mass
  field['xHI'][:] = args.xHI
  field['ye'][:] = args.ye
  field['ie'][:] = args.temperature / cosmology.units.TEMPERATURE
  field['sml'][:] = (mass * 32 / (density * (4./3. * 3.14))) ** 0.333333

  bh = Field(numpoints=1, components={'bhmass':'f4', 'bhmdot':'f4', 'id':'i8'})
  bh['locations'][0,:] = (boxsize * 0.5 + origin)
  bh['id'][:] = -1
  bh['bhmass'][:] = args.bhmass
  bh['bhmdot'][:] = args.bhmdot

  snapshot.C[...] = C
  snapshot.C['boxsize'] = 0
  snapshot.C['N'] = 0
  snapshot.C['Ntot'] = 0
  field.dump_snapshots([snapshot], ptype=0)
  field['locations'] = field['locations'] + (boxsize / nperside * 0.5)[None, :]
  field['id'] = numpy.arange(nperside.prod()) + nperside.prod()
  field.dump_snapshots([snapshot], ptype=1)
  bh.dump_snapshots([snapshot], ptype=5)
  snapshot.save_all()

if __name__ == '__main__':
  main()
