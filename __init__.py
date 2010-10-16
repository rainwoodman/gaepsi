from constant.GADGET import *
import constant.SI as SI
import cosmology
import reader
import field
import readers.hydro3200
import readers.sphray
import snapshot

Field = field.Field
Snapshot = snapshot.Snapshot

Readers = dict(
   hydro3200 = readers.hydro3200.Reader(),
   sphray = readers.sphray.Reader(),
  )
