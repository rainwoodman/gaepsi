from field import Field
from numpy import array


locations = array([[10.0, 10.0, 10.0], [5,5,10]])
sml = array([10.0, 10.0])
value = array([1.0, 1.0])

field = Field(locations = locations, sml = sml, value = value,
            origin=array([0,0,0]), boxsize=[20,20,20], periodical = False)
