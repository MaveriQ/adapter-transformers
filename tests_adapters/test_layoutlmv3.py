import unittest

from transformers import LayoutLMv3Config
from transformers.testing_utils import require_torch

from .methods import (
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
    UniPELTTestMixin,
)
from .test_adapter import AdapterTestBase, make_config
from .test_adapter_backward_compability import CompabilityTestMixin
from .composition.test_parallel import ParallelAdapterInferenceTestMixin, ParallelTrainingMixin
from .test_adapter_conversion import ModelClassConversionTestMixin
from .test_adapter_embeddings import EmbeddingTestMixin
from .test_adapter_fusion_common import AdapterFusionModelTestMixin
from .test_adapter_heads import PredictionHeadModelTestMixin


class LayoutLMv3AdapterTestBase(AdapterTestBase):
    config_class = LayoutLMv3Config
    config = make_config(
        LayoutLMv3Config,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=37,
    )
    tokenizer_name = "microsoft/layoutlmv3-base"


@require_torch
class LayoutLMv3AdapterTest(
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
    UniPELTTestMixin,
    EmbeddingTestMixin,
    AdapterFusionModelTestMixin,
    CompabilityTestMixin,
    PredictionHeadModelTestMixin,
    # ParallelAdapterInferenceTestMixin,
    # ParallelTrainingMixin,
    LayoutLMv3AdapterTestBase,
    unittest.TestCase,
):
    pass


@require_torch
class LayoutLMv3ClassConversionTest(
    ModelClassConversionTestMixin,
    LayoutLMv3AdapterTestBase,
    unittest.TestCase,
):
    pass
