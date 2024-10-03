from tai.findtree import *
from tai.wl import *

if opt.general.enable_wolfram:
    wolf = wolfram()
else:
    wolf = None

os._exit(-1)