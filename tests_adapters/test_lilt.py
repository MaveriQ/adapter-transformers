import unittest

from transformers import LiltConfig, AutoTokenizer
from transformers.testing_utils import require_torch

from .methods import (
    BottleneckAdapterTestMixin,
    UniPELTTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
    
)
from datasets import load_dataset
from .test_adapter import AdapterTestBase, make_config
from .test_adapter_backward_compability import CompabilityTestMixin
from .composition.test_parallel import ParallelAdapterInferenceTestMixin
from .test_adapter_conversion import ModelClassConversionTestMixin
from .test_adapter_fusion_common import AdapterFusionModelTestMixin
from .test_adapter_heads import PredictionHeadModelTestMixin



class LiltAdapterTestBase(AdapterTestBase):
    config_class = LiltConfig
    config = make_config(
        LiltConfig,
        hidden_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        channel_shrink_ratio=1,
        vocab_size=250002,

    )
    tokenizer_name = "SCUT-DLVCLab/lilt-infoxlm-base"

    def dataset(self, tokenizer=None):
        # setup tokenizer
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)#, use_fast=False)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = tokenizer.eos_token
        dataset = load_dataset("nielsr/funsd-layoutlmv3",split='train')
        dataset.set_transform(self.prepare_examples)
        return dataset

    def prepare_examples(self,batch):
        encoding = self.tokenizer(batch["tokens"],
                            boxes=batch["bboxes"],
                            word_labels=batch["ner_tags"],
                            padding="max_length",
                            max_length=128,
                            truncation=True,
                            return_tensors="pt")

        return encoding

@require_torch
class LiltAdapterTest(
    BottleneckAdapterTestMixin,
    # CompacterTestMixin,
    # IA3TestMixin,
    # LoRATestMixin,
    # PrefixTuningTestMixin,
    # UniPELTTestMixin,
    AdapterFusionModelTestMixin,
    CompabilityTestMixin,
    PredictionHeadModelTestMixin,
    # ParallelAdapterInferenceTestMixin,
    LiltAdapterTestBase,
    unittest.TestCase,
):
    pass


@require_torch
class LiltClassConversionTest(
    ModelClassConversionTestMixin,
    LiltAdapterTestBase,
    unittest.TestCase,
):
    pass
