from tests.models.layoutlmv3.test_modeling_layoutlmv3 import *
from transformers import LayoutLMv3AdapterModel
from transformers.testing_utils import require_torch

from .base import AdapterModelTesterMixin


@require_torch
class LayoutLMv3AdapterModelTest(AdapterModelTesterMixin, LayoutLMv3ModelTest):
    all_model_classes = (LayoutLMv3AdapterModel,)
    fx_compatible = False
