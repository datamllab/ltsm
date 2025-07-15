from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from dataclasses import dataclass
import torch

class BaseProcessor:
    def __init__(self):
        pass

    def process(self, raw_data, train_data, val_data, test_data, fit_train_only=False):
        pass

    def inverse_process(self, data):
        pass

    def save(self, save_dir):
        pass

    def load(self, save_dir):
        pass
