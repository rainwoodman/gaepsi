import hydro3200
import d4
import hydro3200iso
import hydro3200tab
import sphray

Readers = dict(
   hydro3200 = hydro3200.Reader(),
   hydro3200iso = hydro3200iso.Reader(),
   hydro3200tab = hydro3200tab.Reader(),
   sphray = sphray.Reader(),
   d4 = d4.Reader(),
  )
