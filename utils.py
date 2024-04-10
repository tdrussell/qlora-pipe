import sys
import os.path
import torch

sys.path.insert(0, os.path.abspath('axolotl/src'))

from axolotl.utils.distributed import is_main_process, zero_first

DTYPE_MAP = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}
