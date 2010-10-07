class FakeSnapShot :
  def push(self): pass
  def load(self, field): pass
  def pop(self): pass
  def __init__(self):
    self.P = [{}]
    self.header = {}
    self.P[0]['pos'] = array([
     [1, 1, 1],
     [1.1, 1.3, 1],
     [3, 3, 1]])
    self.P[0]['mass'] = array([
     1, 1, 1])
    self.P[0]['sml'] = array([0.2,0.2,0.2])
    self.Nparticle = array([3, 0, 0, 0, 0, 0])
    self.header['boxsize'] = 4.0
