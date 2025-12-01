# MML Studio 12: Chemical Reaction Prediction with Transformers

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/digitalmoleculardesign/mml_studio_12/blob/main/reaction_prediction_2025.ipynb)

## Course: 06-731 Molecular Machine Learning
**Carnegie Mellon University - Gomes Group**

---

## Overview

This studio explores **chemical language models** for reaction prediction, treating chemical reactions as a machine translation problem (reactants → products). We use transformer-based models to predict:

- **Forward reaction prediction**: Given reactants, predict products
- **Retrosynthesis**: Given a target product, predict required reactants
- **Atom mapping**: Track how atoms rearrange during reactions

## Background

This notebook is an updated version of the 2022 "Digital Molecular Design Studio" materials from Philippe Schwaller's guest lecture on Chemical Language Models. The original used OpenNMT-py, which is now deprecated.

### What Changed (2022 → 2025)

| Component | 2022 Version | 2025 Version |
|-----------|--------------|---------------|
| **Framework** | OpenNMT-py 2.2.0 | HuggingFace Transformers |
| **Model** | Custom Molecular Transformer | ReactionT5v2 (pre-trained) |
| **Python** | 3.6-3.8 | 3.10-3.12 |
| **Training** | From scratch (24+ hours) | Fine-tuning (minutes) |

See [NOTEBOOK_UPDATE_DOCUMENTATION.md](NOTEBOOK_UPDATE_DOCUMENTATION.md) for detailed change notes.

## Contents

- `reaction_prediction_2025.ipynb` - Main tutorial notebook
- `NOTEBOOK_UPDATE_DOCUMENTATION.md` - Documentation of changes from original
- `requirements.txt` - Python dependencies

## Quick Start

### Option 1: Google Colab (Recommended)

Click the "Open in Colab" badge above. All dependencies install automatically.

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/digitalmoleculardesign/mml_studio_12.git
cd mml_studio_12

# Create conda environment
conda create -n mml_studio_12 python=3.11 -y
conda activate mml_studio_12

# Install dependencies
conda install -c conda-forge rdkit -y
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook reaction_prediction_2025.ipynb
```

## Topics Covered

### Part 1: Foundations
- SMILES representation of molecules and reactions
- Tokenization strategies for chemical language models
- Canonicalization and data preprocessing

### Part 2: Reaction Prediction
- Loading pre-trained ReactionT5v2 models
- Forward reaction prediction (reactants → products)
- Retrosynthesis prediction (products → reactants)
- Evaluation metrics (Top-1, Top-5 accuracy)

### Part 3: Advanced Topics
- Atom mapping with RXNMapper
- IBM RXN for Chemistry API
- Data augmentation via SMILES randomization
- Fine-tuning on custom datasets

### Part 4: Analysis
- Visualization of chemical reactions
- Error analysis and debugging
- Common failure patterns

## Key Models and Tools

- **[ReactionT5v2](https://github.com/sagawatatsuya/ReactionT5v2)** - T5-based model pre-trained on Open Reaction Database
- **[RXNMapper](https://github.com/rxn4chemistry/rxnmapper)** - Attention-based atom mapping
- **[rxn4chemistry](https://github.com/rxn4chemistry/rxn4chemistry)** - IBM RXN API client
- **[RDKit](https://www.rdkit.org/)** - Cheminformatics toolkit

## References

### Primary Papers

1. Schwaller, P. et al. "Molecular Transformer: A Model for Uncertainty-Calibrated Chemical Reaction Prediction" *ACS Central Science* (2019) [DOI](https://doi.org/10.1021/acscentsci.9b00576)

2. Sagawa, T. & Kojima, R. "ReactionT5: a pre-trained transformer model for accurate chemical reaction prediction with limited data" *Journal of Cheminformatics* (2025) [DOI](https://doi.org/10.1186/s13321-025-01075-4)

3. Schwaller, P. et al. "Extraction of organic chemistry grammar from unsupervised learning of chemical reactions" *Science Advances* (2021) [DOI](https://doi.org/10.1126/sciadv.abe4166)

### Review

- Schwaller, P. "Machine Intelligence for Chemical Reaction Space" *WIREs Computational Molecular Science* (2022) [Link](https://wires.onlinelibrary.wiley.com/doi/full/10.1002/wcms.1604)

## Original Source

This notebook is adapted from: [schwallergroup/dmds_language_models_for_reactions](https://github.com/schwallergroup/dmds_language_models_for_reactions)

## License

MIT License

---

**Gomes Group** | Carnegie Mellon University | [gomesgroup.github.io](https://gomesgroup.github.io)
