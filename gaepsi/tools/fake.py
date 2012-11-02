from gaepsi.field import Field
from numpy import array


field = Field(numpoints=1, components={'mass':'f4', 'vel':('f4', 3), 'sml':'f4'}, boxsize=20)

field['locations'][0,:] = array([10.0,10.0,10.0])
field['vel'][0,:] = array([10.0,20.0,30.0])
field['sml'][0] = 10.0
field['mass'][0] = 1.0


