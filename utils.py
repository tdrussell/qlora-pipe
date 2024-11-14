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

def eta_str(eta):
    eta = int(eta)
    if eta > 3600:
        return f'{eta // 3600}h{(eta % 3600) // 60}m'
    return f'{eta // 60}m{eta % 60}s' if eta > 60 else f'{eta}s'

def count_str(num):
    num = int(num)
    if num > 1000000000:
        return f"{num/1000000000:.2f}G"
    if num > 1000000:
        return f"{num/1000000:.2f}M"
    elif num > 1000:
        return f"{num/1000:.2f}k"
    return str(num)
