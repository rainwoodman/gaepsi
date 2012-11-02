raise Exception("The module threads is deprecated")
import threading
from Queue import Queue, Empty
import sys

def work(jobfunc, jobs, num_workers=8):
  err_q = Queue()
  if not isinstance(jobs, Queue):
    job_q = Queue()
    for job in jobs:
      job_q.put(job)
  else:
    job_q = jobs

  for i in range(num_workers):
    thread = threading.Thread(target=jobfunc, args=[job_q, err_q])
    thread.daemon = False
    thread.start()

  job_q.join()
  if not err_q.empty() : 
    e = err_q.get()
    raise e[0], e[1], e[2]

def job_with_lock(func):
  lock = threading.RLock()
  def wrapped(queue, error):
    while True:
      try:
        args = queue.get(block=False)
      except Empty:
        break 
      if type(args) is list:
        args = args
        kwargs = {}
      if type(args) is dict:
        kwargs = args
        args = []
      kwargs['lock'] = lock
      try:
        func(*args, **kwargs)
      except Exception as err:
        error.put(sys.exc_info())
      finally:
        queue.task_done()
  return wrapped


def job(func):
  def wrapped(queue, error):
    while True:
      try:
        args = queue.get(block=False)
      except Empty:
        break 
      if type(args) is list:
        args = args
        kwargs = {}
      elif type(args) is dict:
        kwargs = args
        args = []
      else:
        args = [args]
        kwargs= {}
      try:
        func(*args, **kwargs)
      except Exception as err:
        error.put(sys.exc_info())
      finally:
        queue.task_done()
  return wrapped

