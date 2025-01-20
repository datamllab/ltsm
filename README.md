# Understanding Different Design Choices in Training Large Time Series Models
<img width="700" height="290" src="./imgs/ltsm_model.png">

This work investigates the transition from traditional Time Series Forecasting (TSF) to Large Time Series Models (LTSMs), leveraging universal transformer-based models. Training LTSMs on diverse time series data introduces challenges due to varying frequencies, dimensions, and patterns. We explore various design choices for LTSMs, including pre-processing, model configurations, and dataset setups. We introduce **Time Series Prompt**, a statistical prompting strategy, and $\texttt{LTSM-bundle}$, which encapsulates the most effective design practices identified.


## Why LTSM-bundle?
The LTSM-bundle package leverages the HuggingFace transformers toolkit, offering flexibility to switch between different advanced language models as the backbone. It is easy to tailor the general LTSMs to their specific time series forecasting needs by selecting the most suitable language model from a wide array of options. The flexibility enhances the adaptability of the package across different industries and data types, ensuring optimal performance in diverse scenarios.

## Installation
```
conda create -n ltsm python=3.8.0
conda activate ltsm
git clone git@github.com:daochenzha/ltsm.git
cd ltsm
pip3 install -e .
pip3 install -r requirements.txt
```

## Quick Exploration on LTSM-bundle 

Training on **[Time Series Prompt]** and **[Linear Tokenization]**
```bash
bash scripts/train_ltsm_csv.sh
```

Training on **[Text Prompt]** and **[Linear Tokenization]**
```bash
bash scripts/train_ltsm_textprompt_csv.sh
```

Training on **[Time Series Prompt]** and **[Time Series Tokenization]**
```bash
bash scripts/train_ltsm_tokenizer_csv.sh
```

