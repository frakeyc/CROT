## Efficient High-Dimensional Time Series Forecasting with Transformers: A Channel Reordering Perspective

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Framework-Accelerate-yellow.svg)](https://huggingface.co/docs/accelerate)



## 🚀 Quick Start

### Installation

```bash
# Method 1: Using pip
pip install -r requirements.txt

# Method 2: Using conda (recommended)
conda env create -f environment.yaml
conda activate tsf

# Install optional dependencies for full functionality
pip install pandas torchinfo torchsort
```

### Usage

```bash
# 🖥️ Single GPU training
accelerate launch --num_processes=1 run.py --model CROT --data "Measles" --gpu 0

# 🚀 Multi-GPU training (auto-detect all GPUs)
accelerate launch run.py --model CROT --data "Measles"

# 🎯 Specific GPU selection (e.g. 4 GPUs, id: 0,2,3,7)
accelerate launch --num_processes=4 run.py --model CROT --data "Measles" --gpu 0,2,3,7

# 📋 List available models
accelerate launch run.py --list-models

# ℹ️ Show framework information
python run.py --info
```

## 🔧 Configuration System

### Model Configuration

Create dataset-specific configurations in `configs/`:

```yaml
# configs/UCast.yaml
Measles:
  enc_in: 1161
  train_epochs: 10
  alpha: 0.01
  seq_len_factor: 4
  learning_rate: 0.001

Air_Quality:
  enc_in: 2994
  train_epochs: 15
  alpha: 0.1
  seq_len_factor: 5
  learning_rate: 0.0001
```

## 🏗️ Architecture Overview

```
📁 CROT Model
├── 🚀 run.py                     # Main entry point with GPU management
├── 🏗️  core/                     # Core framework components
│   ├── 📝 config/                # Configuration management system
│   │   ├── base.py               # Base configuration classes
│   │   ├── manager.py            # Configuration manager
│   │   └── model_configs.py      # Model-specific configs
│   ├── 📊 registry/              # Model/dataset registration
│   │   ├── __init__.py           # Registry decorators
│   │   └── model_registry.py     # Model registration system
│   ├── 🤖 models/                # Model management and loading
│   │   ├── model_manager.py      # Dynamic model loading
│   │   └── __init__.py           # Model manager interface
│   ├── 📊 data/                  # Self-contained data pipeline
│   │   ├── data_provider.py      # Main data provider
│   │   ├── data_factory.py       # Dataset factory
│   │   └── data_loader.py        # Custom dataset classes
│   ├── 🧪 experiments/           # Experiment orchestration
│   │   ├── base_experiment.py    # Base experiment class
│   │   └── long_term_forecasting.py  # Forecasting experiments
│   ├── ⚙️  execution/             # Execution engine
│   │   └── runner.py             # Experiment runners
│   ├── 🛠️  utils/                # Self-contained utilities
│   │   ├── tools.py              # Training utilities
│   │   ├── metrics.py            # Evaluation metrics
│   │   ├── timefeatures.py       # Time feature extraction
│   │   ├── augmentation.py       # Data augmentation
│   │   ├── masked_attention.py   # Attention mechanisms
│   │   └── masking.py            # Masking utilities
│   ├── 🔌 plugins/               # Plugin system for extensibility
│   └── 💻 cli/                   # Command-line interface
│       └── argument_parser.py    # Comprehensive CLI parser
├── 🤖 models/                    # Model implementations with @register_model
│   ├── CROT.py                  # High-dimensional specialist
│   ├── TimesNet.py               # 2D temporal modeling
│   ├── iTransformer.py           # Inverted transformer
│   ├── ModernTCN.py              # Modern TCN
│   └── ...                       # 16+ other models
├── 🗂️ configs/                   # Model-dataset configurations
├── 🔍 config_hp/                 # Hyperparameter search configs
├── 🧱 layers/                    # Neural network building blocks
└── 📊 results/                   # Experiment outputs and logs
```

## 📈 Performance Benchmarks
<p align="center">
<img src=".\pics\results.png" height = "300" alt="" align=center />
</p>

