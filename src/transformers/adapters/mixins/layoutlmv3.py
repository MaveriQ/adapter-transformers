import logging
from typing import Iterable, Tuple

import torch.nn as nn

from ..layer import AdapterLayer
from ..model_mixin import (
    EmbeddingAdaptersMixin,
    EmbeddingAdaptersWrapperMixin,
    InvertibleAdaptersMixin,
    ModelAdaptersMixin,
    ModelWithHeadsAdaptersMixin,
)


logger = logging.getLogger(__name__)


# For backwards compatibility, LayoutLMv3SelfOutput inherits directly from AdapterLayer
class LayoutLMv3SelfOutputAdaptersMixin(AdapterLayer):
    """Adds adapters to the LayoutLMv3SelfOutput module."""

    def __init__(self):
        super().__init__("mh_adapter", None)


# For backwards compatibility, LayoutLMv3Output inherits directly from AdapterLayer
class LayoutLMv3OutputAdaptersMixin(AdapterLayer):
    """Adds adapters to the LayoutLMv3Output module."""

    def __init__(self):
        super().__init__(f"output_adapter", None)


class LayoutLMv3ModelAdaptersMixin(EmbeddingAdaptersMixin, InvertibleAdaptersMixin, ModelAdaptersMixin):
    """Adds adapters to the LayoutLMv3Model module."""

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in enumerate(self.encoder.layer):
            yield i, layer


class LayoutLMv3ModelWithHeadsAdaptersMixin(EmbeddingAdaptersWrapperMixin, ModelWithHeadsAdaptersMixin):
    pass
