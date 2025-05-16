# VisualSphinx-Generator

This is the official repository for paper "[VisualSphinx: Large-Scale Synthetic Vision Logic Puzzles for RL]". 

## Installation

**Build environment**
```
git clone https://github.com/VisualSphinx/VisualSphinx-Generator.git
cd VisualSphinx-Generator
conda create -n VisualSphinx python=3.10 -y
conda activate VisualSphinx
pip install -r requirements.txt
```

## Generate Data
Please go into [pipeline](/pipeline) for reproduce VisualSphinx. Please do not forget to define your API-Keys in [api_config.py](pipeline/api_config.py).
