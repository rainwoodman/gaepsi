from time import time as default_timer

timer = 0
def reset():
  global timer
  timer = default_timer()
  return 0
def read():
  global timer
  return default_timer()-timer
def restart():
  global timer
  rt = default_timer() - timer
  timer = default_timer()
  return rt

