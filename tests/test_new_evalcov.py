import gvar
from gvar import _utilities
from gvar import _evalcov_fast

evalcov_blocks = _evalcov_fast.evalcov_blocks
gvar.evalcov_blocks = evalcov_blocks
_utilities.evalcov_blocks = evalcov_blocks

from .test_bufferdict import *
from .test_gvar_dataset import *
from .test_gvar_other import *
from .test_gvar import *
