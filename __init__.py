from constant.GADGET import *
import constant.SI as SI
import cosmology
import reader
import field
import readers.hydro3200
import readers.d4
import readers.hydro3200iso
import readers.sphray
import snapshot
import snapdir

Field = field.Field
Snapshot = snapshot.Snapshot
Snapdir = snapdir.Snapdir

Readers = dict(
   hydro3200 = readers.hydro3200.Reader(),
   hydro3200iso = readers.hydro3200iso.Reader(),
   sphray = readers.sphray.Reader(),
   d4 = readers.d4.Reader(),
  )
