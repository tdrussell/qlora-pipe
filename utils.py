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

def utfplot(eval_loss):
    try:
        import plotille
    except ImportError:
        # Skipping plots
        return

    # Create the plot
    fig = plotille.Figure()
    fig.width = 60
    fig.height = 20
    fig.set_x_limits(min_=0)
    fig.x_label = 'Epoch'
    fig.y_label = 'Evaluation Loss'

    for i in range(1, len(eval_loss)):
        color = "red" if eval_loss[i] > eval_loss[i - 1] else "green"
        fig.plot([i-1,i], eval_loss[i-1:i+1], lc=color)

    # Print the plot
    print(fig.show())
