import sys
import os.path

sys.path.insert(0, os.path.abspath('axolotl/src'))

from axolotl.utils.distributed import is_main_process, zero_first