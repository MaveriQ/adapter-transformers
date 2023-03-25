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

from transformers.testing_utils import torch_device
from PIL import Image
from transformers import LayoutLMv3FeatureExtractor
import torch
class LayoutLMv3AdapterTestBase(AdapterTestBase):
    config_class = LayoutLMv3Config
    config = make_config(
        LayoutLMv3Config,
        batch_size=2,
        num_channels=3,
        image_size=4,
        patch_size=2,
        text_seq_length=7,
        vocab_size=99,
        hidden_size=36,
        num_hidden_layers=3,
        num_attention_heads=4,
        intermediate_size=37,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        coordinate_size=6,
        shape_size=6,
        num_labels=3,
        num_choices=4,
        scope=None,
        range_bbox=1000,

    )
    tokenizer_name = "microsoft/layoutlmv3-base"

    def get_input_samples(self, shape=None, vocab_size=5000, config=None):
        
        image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False)
        pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
        input_ids = torch.tensor([[1, 2]])
        bbox = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]).unsqueeze(0)
        attention_mask = torch.tensor([[1, 1]])

        return {"pixel_values": pixel_values.to(torch_device),
                "input_ids": input_ids.to(torch_device),
                "bbox": bbox.to(torch_device),
                "attention_mask": attention_mask.to(torch_device)}
    
    def dataset(self, tokenizer=None):
        return super().dataset(tokenizer)

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
