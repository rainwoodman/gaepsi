mpicomm = None
try:
  from mpi4py import MPI
  def mpi_timer():
    return MPI.Wtime()
  default_timer = mpi_timer
  mpicomm = MPI.COMM_WORLD
  if mpicomm.rank == 0: print 'using MPI timer', 'precision',  MPI.Wtick()
except:
  from time import time as default_timer

programstart = default_timer()

class Session:
  def __init__(self, descr):
    self.descr = descr
    self.start_time = 0
    self.checkpoint_time = 0
  def start(self):
    if mpicomm!=None:
      mpicomm.Barrier()
    self.start_time = default_timer()
    self.checkpoint_time = self.start_time
    if mpicomm == None or mpicomm.rank == 0:
      print self.descr, 'started at', self.start_time - programstart
  def checkpoint(self, msg):
    if mpicomm!=None:
      mpicomm.Barrier()
    rt = default_timer() - self.checkpoint_time
    self.checkpoint_time = default_timer()
    if mpicomm == None or mpicomm.rank == 0:
      print self.descr, msg, 'used', rt
    return rt
    
  def end(self):
    if mpicomm!=None:
      mpicomm.Barrier()
    newtimer = default_timer()
    if mpicomm == None or mpicomm.rank == 0:
      print self.descr, 'ended at', newtimer - programstart, 'time used', newtimer - self.start_time

def session(descr):
  s = Session(descr)
  s.start()
  return s

