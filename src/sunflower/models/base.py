from abc import abstractmethod
from typing import Any

import torch
import torch.nn as nn

from PIL import Image


class AIModel(nn.Module):
    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError(
            f"Please implement forward method for {type(self).__name__}"
        )
