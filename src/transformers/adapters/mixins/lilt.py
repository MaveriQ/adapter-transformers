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


# For backwards compatibility, BertSelfOutput inherits directly from AdapterLayer
class LiltSelfOutputAdaptersMixin(AdapterLayer):
    """Adds adapters to the BertSelfOutput module."""

    def __init__(self):
        super().__init__("mh_adapter", None)


# For backwards compatibility, BertOutput inherits directly from AdapterLayer
class LiltOutputAdaptersMixin(AdapterLayer):
    """Adds adapters to the BertOutput module."""

    def __init__(self, modality,config):
        super().__init__(f"{modality}_output_adapter", config)


class LiltModelAdaptersMixin(EmbeddingAdaptersMixin, InvertibleAdaptersMixin, ModelAdaptersMixin):
    """Adds adapters to the BertModel module."""

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in enumerate(self.encoder.layer):
            yield i, layer


class LiltModelWithHeadsAdaptersMixin(EmbeddingAdaptersWrapperMixin, ModelWithHeadsAdaptersMixin):
    pass