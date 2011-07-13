import threading
from Queue import Queue
import sys

def work(job, job_q, num_workers=8):
  err_q = Queue()
  for i in range(num_workers):
    thread = threading.Thread(target=job, args=[job_q, err_q])
    thread.daemon = True
    thread.start()
  job_q.join()
  if not err_q.empty() : 
    e = err_q.get()
    raise e[0], e[1], e[2]

def job(func):
  lock = threading.RLock()
  def wrapped(queue, error):
    while True:
      args = queue.get()
      args = list(args) + [lock]
      try:
        func(*args)
      except Exception as err:
        error.put(sys.exc_info())
      finally:
        queue.task_done()
  return wrapped

