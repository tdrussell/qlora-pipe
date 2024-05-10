from datetime import datetime
import sys
import os.path
import torch

sys.path.insert(0, os.path.abspath('axolotl/src'))

from axolotl.utils.distributed import is_main_process, zero_first  # type: ignore # noqa

DTYPE_MAP = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}

# Simplified logger-like printer.
def log(msg):
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]}] [INFO] [qlora-pipe] {msg}')
