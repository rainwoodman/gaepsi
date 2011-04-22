import hydro3200
import d4
import hydro3200iso
import hydro3200tab
import sphray
import sphray2
import bluedrop

Readers = dict(
   hydro3200 = hydro3200.Reader(),
   hydro3200iso = hydro3200iso.Reader(),
   hydro3200tab = hydro3200tab.Reader(),
   sphray = sphray.Reader(),
   sphray2 = sphray2.Reader(),
   d4 = d4.Reader(),
   bluedrop = bluedrop.Reader(),
  )
