import _gadgetbase
class constants:
      OmegaB = 0.044
      flag_sfr = 1
      flag_sft = 1
      flag_met = 1
      flag_feedback = 1
      flag_cool = 1
      flag_entropy = 0
      flag_double = 0
      flag_ic_info = 0

Snapshot, GroupTab, SubHaloTab = _gadgetbase.Snapshot(
     idtype='u4',
     floattype='f4',
     constants=constants)
