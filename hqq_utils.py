from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import transformers
from hqq.core import quantize as hqq_quantize
from torch import nn

import peft
from utils import DTYPE_MAP


# Monkeypatch PEFT so that target_modules='all-linear' targets the HQQLinear layers, which are not
# subclasses of nn.Linear, unlike BNB.
def _maybe_include_all_linear_layers(peft_config: peft.PeftConfig, model: nn.Module) -> peft.PeftConfig:
    """
    Helper function to update `target_modules` to all linear/Conv1D layers if provided as 'all-linear'. Adapted from
    the QLoRA repository: https://github.com/artidoro/qlora/blob/main/qlora.py
    """

    # if `target_modules` is a string, convert to lower case and check if it matches "all-linear"
    if not (
        isinstance(peft_config.target_modules, str)
        and peft_config.target_modules.lower() == peft.tuners.tuners_utils.INCLUDE_LINEAR_LAYERS_SHORTHAND
    ):
        return peft_config

    if not isinstance(model, transformers.PreTrainedModel):
        raise ValueError(
            f'Only instances of PreTrainedModel support `target_modules={peft.tuners.tuners_utils.INCLUDE_LINEAR_LAYERS_SHORTHAND!r}`'
        )

    # add HQQLinear
    linear_classes = (torch.nn.Linear, transformers.pytorch_utils.Conv1D, hqq_quantize.HQQLinear)

    linear_module_names = set()
    for name, module in model.named_modules():
        # match with all linear classes.
        if isinstance(module, linear_classes):
            names = name.rsplit('.', 1)[-1]  # get the base name
            linear_module_names.add(names)

    # ignore the last classification head for text generation models
    output_emb = model.get_output_embeddings()
    if output_emb is not None:
        last_module_name = [name for name, module in model.named_modules() if module is output_emb][0]
        linear_module_names -= {last_module_name}
    peft_config.target_modules = linear_module_names
    return peft_config

peft.tuners.tuners_utils._maybe_include_all_linear_layers = _maybe_include_all_linear_layers


@dataclass
class CustomHQQConfig:
    nbits: int = 4
    group_size: int = 64
    view_as_float: bool = False
    axis: int = 0
    dynamic_config: dict[str, Any] = field(default_factory=dict)
    skip_modules: list[str] = field(default_factory=lambda: ['lm_head'])
    compute_dtype: str = 'float32'

    def __post_init__(self):
        self.compute_dtype = DTYPE_MAP[self.compute_dtype]

    def use_aten(self):
        return self.axis == 0 and all(d.get('axis', self.axis) == 0 for d in self.dynamic_config.values())

    def get_dict(self, full_name):
        """Get final config dict to use for quantization, for module with full_name."""
        kwargs = asdict(self)
        kwargs.pop('compute_dtype')
        kwargs.pop('skip_modules')
        dynamic_config = kwargs.pop('dynamic_config')
        for key, value in dynamic_config.items():
            if key in full_name:
                kwargs.update(value)
                break
        return hqq_quantize.BaseQuantizeConfig(**kwargs)
