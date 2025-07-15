# LTSM-Bundle: A Toolbox and Benchmark on Large Language Models for Time Series Forecasting

<div align="center">
<img src="./imgs/ltsm_model.png" width="700" height="290" alt="LTSM Model">
</div>

[![Test](https://github.com/daochenzha/ltsm/actions/workflows/test.yml/badge.svg)](https://github.com/daochenzha/ltsm/actions/workflows/test.yml)

> Empowering forecasts with precision and efficiency.

## Table of Contents

* [Overview](#overview)
* [Why LTSM-bundle](#why-ltsm-bundle)
* [Features](#features)
* [Installation](#installation)
* [Quick Start](#quick-start)
* [Project Structure](#project-structure)
* [Datasets and Prompts](#datasets-and-prompts)
* [Model Access](#model-access)
* [Cite This Work](#cite-this-work)
* [License](#license)
* [Acknowledgments](#acknowledgments)

---

## Overview

This work investigates the transition from traditional Time Series Forecasting (TSF) to Large Time Series Models (LTSMs), leveraging large transformer-based models like GPT. Training LTSMs on diverse time series data introduces challenges due to varying frequencies, dimensions, and patterns.

We explore multiple design choices, including pre-processing strategies, tokenization, model architectures, and dataset setups. We introduce:

* **Time Series Prompt**: A statistical prompting strategy
* **LTSM-bundle**: A toolkit encapsulating effective design practices

The project is developed by the [Data Lab at Rice University](https://cs.rice.edu/~xh37/).

---

## Why LTSM-bundle?

The LTSM-bundle leverages HuggingFace transformers, allowing flexible integration of large-scale pre-trained language models for time series tasks. Users can customize the pipeline to fit specific forecasting needs with minimal overhead, making it adaptable across various domains and industries.

Key highlights:

* Plug-and-play with GPT-style backbones
* Modular pipeline for easy experimentation
* Support for statistical and text prompts

---

## Features

| Category          | Highlights                                                                                                                          |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| ⚙ï¸ Architecture | Modular design, GPT-style transformers for time series                                                                              |
| 📝 Prompting      | Time Series Prompt & Text Prompt support                                                                                            |
| ⚡️ Performance    | GPU acceleration, optimized pipelines                                                                                               |
| 🔧 Integrations   | LoRA support, JSON/CSV-based dataset and prompt interfaces                                                                          |
| 🔬 Testing        | Unit and integration tests, GitHub Actions CI                                                                                       |
| 📊 Data           | Built-in data loaders, scalers, and tokenizers                                                                                      |
| 📂 Documentation  | Tutorials in [English](https://github.com/daochenzha/ltsm/tree/main/tutorial) and [Chinese](https://zhuanlan.zhihu.com/p/708804309) |

---

## Installation

We recommend using Conda:

```bash
conda create -n ltsm python=3.8.0
conda activate ltsm
```

Then install the package:

```bash
git clone https://github.com/datamllab/ltsm.git
cd ltsm
pip install -e .
pip install -r requirements.txt
```

---

## 🔧 Training Examples
<!-- Joshua please helps this part -->
```Python
```


## 🔍 Inference Examples

```Python
import os
import torch
import pandas as pd
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from ltsm.models import LTSMConfig, ltsm_model

# Download model config and weights from Hugging Face
config_path  = hf_hub_download("LSC2204/LTSM-bundle", "config.json")
weights_path = hf_hub_download("LSC2204/LTSM-bundle", "model.safetensors")

# Load model and weights
model_config = LTSMConfig()
model_config.load(config_path)
model = ltsm_model.LTSM(model_config)

state_dict = load_file(weights_path)
model.load_state_dict(state_dict)

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

# Load your dataset (e.g., weather)
df_weather = pd.read_csv("/path/to/dataset.csv")
print("Loaded data shape:", df_weather.shape)

# Load prompts per feature
feature_prompts = {}
prompt_dir = "/path/to/prompts/"
for feature, filename in {
    "T (degC)": "weather_T (degC)_prompt.pth.tar",
    "rain (mm)": "weather_rain (mm)_prompt.pth.tar"
}.items():
    prompt_tensor = torch.load(os.path.join(prompt_dir, filename))
    feature_prompts[feature] = prompt_tensor.squeeze(0).float().to(device)

# Predict (custom code here depending on your model usage)
# For example:
with torch.no_grad():
    inputs = feature_prompts["T (degC)"].unsqueeze(0)
    preds = model(inputs)
    print("Prediction output shape:", preds.shape)
```

---

## Project Structure

```text
└── ltsm-package/
    ├── datasets
    │   └── README.md
    ├── imgs
    │   ├── ltsm_model.png
    │   ├── prompt_csv_tsne.png
    │   └── stat_prompt.png
    ├── ltsm
    │   ├── common                  # Base classes 
    │   ├── data_pipeline           # Model lifecycle management and training pipeline
    │   ├── data_provider           # Dataset construction
    │   ├── data_reader             # Read input data from various formats (CSV, JSON, etc.)
    │   ├── evaluate_pipeline       # Evaluation workflow for model performance
    │   ├── layers                  # Custom neural network components
    │   ├── models                  # Implementations: LTSM, DLinear, Informer, PatchTST
    │   ├── prompt_reader           # Prompt generation and formatting
    │   ├── sk_interface            # Scikit-learn style interface
    │   └── utils                   # Shared helper functions
    ├── multi_agents_pipeline       # Multi-agent time series reasoning framework
    │   ├── Readme.md
    │   ├── agents                  # Agent definitions: Planner, QA, TS, Reward
    │   ├── llm-server.py           # Local LLM server interface
    │   ├── ltsm_inference.py       # Inference script using LTSM pipeline
    │   ├── main.py                 # Pipeline entry point
    │   └── model_config.yaml       # Configuration file for models and agents
    ├── requirements.txt
    ├── setup.py
    ├── tests                       # Unit tests for LTSM modules
    │   ├── common
    │   ├── data_pipeline
    │   ├── data_provider
    │   ├── data_reader
    │   ├── evaluate_pipeline
    │   ├── models
    │   └── test_scripts
    └── tutorial
        └── README.md
```

---

## Datasets and Prompts

Download datasets:

```bash
cd datasets
# Google Drive link:
https://drive.google.com/drive/folders/1hLFbz0FRxdiDCzgFYtKCOPJYSBVvwW9P
```

Download time series prompts:

```bash
cd prompt_bank/prompt_data_csv
# Same Google Drive link applies
```

---

## Model Access

You can find our trained LTSM models on Hugging Face:

➡️ [https://huggingface.co/LSC2204/LTSM-bundle](https://huggingface.co/LSC2204/LTSM-bundle)

---

## Cite This Work

If you find this work useful, please cite:

```bibtex
@misc{chuang2025ltsmbundletoolboxbenchmarklarge,
      title={LTSM-Bundle: A Toolbox and Benchmark on Large Language Models for Time Series Forecasting},
      author={Yu-Neng Chuang and Songchen Li and Jiayi Yuan and Guanchu Wang and Kwei-Herng Lai and Songyuan Sui and Leisheng Yu and Sirui Ding and Chia-Yuan Chang and Qiaoyu Tan and Daochen Zha and Xia Hu},
      year={2025},
      eprint={2406.14045},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2406.14045},
}
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](https://choosealicense.com/licenses/mit/) file for details.

---

## Acknowledgments

We thank all contributors and collaborators involved in the LTSM project. Special thanks to the Data Lab at Rice University and the open-source community for enabling fast prototyping and reproducible research.

---

<div align="right">
    <a href="#top">⬆️ Back to Top</a>
</div>
