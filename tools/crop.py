from gaepsi.field import Field, Cut
from gaepsi.snapshot import Snapshot
from gaepsi.tools.meshmap import Meshmap
from numpy import fromstring, newaxis

def crop_snapshot(center, size, map, format, snapname, output):
  cut = Cut(center = center, size=size)
  mesh = Meshmap(map)

  gas = Field(components={'mass':'f4', 'rho':'f4', 'ie':'f4', 'xHI':'f4', 'ye':'f4', 'id':'u8', 'sfr':'f4', 'met':'f4', 'sml':'f4'})
  bh = Field(components={'mass':'f4', 'bhmdot':'f4', 'bhmass':'f4', 'id':'u8'})
  star = Field(components={'mass':'f4', 'sft':'f4', 'met':'f4', 'id':'u8'})

  fids = mesh.cut2fid(cut)
  print fids
  snapshots = [Snapshot(snapname % fid, format) for fid in fids]

  gas.take_snapshots(snapshots, 0, cut=cut)
  star.take_snapshots(snapshots, 4, cut=cut)
  bh.take_snapshots(snapshots, 5, cut=cut)
  left = center - size * 0.5
  gas['locations'][:, :] -= left[newaxis, :]
  star['locations'][:, :] -= left[newaxis, :]
  bh['locations'][:, :] -= left[newaxis, :]
  print bh['id']
  out = Snapshot(output, format, create=True)
  out.header['redshift'] = snapshots[0].header['redshift']
  out.header['time'] = snapshots[0].header['time']
  out.header['unused'][0:3] = center
  out.header['boxsize'] = size
  gas.dump_snapshots([out], 0)
  star.dump_snapshots([out], 4)
  bh.dump_snapshots([out], 5)
