from enum import Enum

class FairnessMethods(Enum):
  RW = 'rw'
  DIR = 'dir'
  DEMV = 'demv'
  EG = 'eg'
  GRID = 'grid'
  AD = 'ad'
  GERRY = 'gerry_fair'
  META = 'meta_fair'
  PREJ = 'prej'
  EOP = 'eop'
  REJ = 'rej_opt'
  NO_ONE = 'no_one'