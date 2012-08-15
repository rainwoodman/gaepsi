import SI
class Units:
  def __init__(self, h):
    C = 2.99792458e5
    G = 43007.1
    H0 = 0.1
    pi = 3.1415926
    #internal units
    TIME = H0 / (SI.H0 * h)
    LENGTH = TIME * (SI.C / C)
    MASS = G / SI.G * LENGTH ** 3 * TIME ** -2
    TEMPERATURE = 1.0
    ENERGY = MASS * (LENGTH / TIME) ** 2
    POWER = ENERGY / TIME

    #quantities
    BOLTZMANN = SI.BOLTZMANN / ((LENGTH / TIME) ** 2 * MASS)
    SOLARMASS = SI.SOLARMASS / MASS
    SOLARLUMINOSITY = SI.SOLARLUMINOSITY / POWER
    LSUN = SOLARLUMINOSITY
    PROTONMASS = SI.PROTONMASS / MASS
    PLANCK = SI.PLANCK / (ENERGY * TIME)

    KPC_h = SI.KPC / LENGTH / h
    MPC_h = KPC_h * 1000
    MYEAR_h = SI.MYEAR / TIME / h
    KPC = SI.KPC / LENGTH
    MPC = KPC * 1000
    METER = SI.METER / LENGTH
    ANGSTROM = 1e-10 * METER
    CM = 1e-2 * METER
    NANOMETER = SI.NANOMETER / LENGTH
    MYEAR = SI.MYEAR / TIME
    YEAR = 1e-6 * MYEAR
    LYMAN_ALPHA_CROSSSECTION = SI.LYMAN_ALPHA_CROSSSECTION / (LENGTH**2)
    J = SI.J / ENERGY
    W = SI.W / (ENERGY / TIME)
    EV = SI.EV / ENERGY
    ERG = SI.ERG / ENERGY
    SECOND = SI.SECOND / TIME
    HERTZ = 1 / SECOND
    RYDBERG = SI.RYDBERG / ENERGY
    CRITICAL_DENSITY = 3 * H0 ** 2/ (8 * pi * G)

    self.set_dict(locals());

  def set_dict(self, locals):
    for field in locals:
      if field == "self": continue
      self.__dict__[field] = locals[field]
  def __str__(self):
    return str(self.__dict__)

