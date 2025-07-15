from dataclasses import dataclass
from transformers import PretrainedConfig
import json

@dataclass
class LTSMConfig(PretrainedConfig):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def load(self, json_file):

        with open(json_file) as f:
            config = json.load(f)

        for key, value in config.items():
            setattr(self, key, value)

        return self