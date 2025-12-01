# Molecular Transformer Reaction Prediction Notebook Update

## Summary of Changes (2022 to 2025)

### Overview

The original notebook (`test_reaction_prediction.ipynb`) from April 2022 used OpenNMT-py for training and inference with the Molecular Transformer. This approach is now deprecated and broken. The updated notebook (`reaction_prediction_2025.ipynb`) uses a modern HuggingFace-based approach with ReactionT5v2.

---

## Key Changes

### 1. Framework Migration

| Aspect | Original (2022) | Updated (2025) |
|--------|-----------------|----------------|
| **Core Framework** | OpenNMT-py 2.2.0 | HuggingFace Transformers 4.40+ |
| **Model** | Custom Molecular Transformer | ReactionT5v2 (pre-trained) |
| **Tokenization** | pyonmttok | HuggingFace AutoTokenizer |
| **Training** | 24+ hours from scratch | Minutes for fine-tuning |
| **Python Version** | 3.6-3.8 | 3.10-3.12 |

### 2. Why OpenNMT-py No Longer Works

1. **Maintenance Mode**: OpenNMT-py announced it's "no longer actively supported" in July 2024
   - Successor project: [Eole](https://github.com/eole-nlp/eole)
   - Forum announcement: https://forum.opennmt.net/t/opennmt-py-is-now-in-maintenance-mode-use-eole-instead/5792

2. **pyonmttok Dependency Failure**:
   - The tokenizer `pyonmttok<2,>=1.23` has no matching distribution for Python 3.12
   - Google Colab now uses Python 3.12 by default
   - Issue: https://github.com/OpenNMT/Tokenizer/issues/329

3. **torchtext Removal**:
   - OpenNMT-py relied on torchtext, which has been deprecated
   - Latest OpenNMT-py versions removed torchtext but introduced new issues

### 3. Solution: ReactionT5v2

**ReactionT5v2** is a T5-based model pre-trained on the Open Reaction Database (ORD):

- **Repository**: https://github.com/sagawatatsuya/ReactionT5v2
- **HuggingFace**: https://huggingface.co/collections/sagawa/reactiont5
- **Paper**: Journal of Cheminformatics (2025) - DOI: 10.1186/s13321-025-01075-4
- **Colab Demo**: https://colab.research.google.com/drive/10Hkx8YJTG5JGXj73XfnEYxZ8fDnsJRcs

**Advantages**:
- Pre-trained on 2M+ reactions from ORD
- State-of-the-art performance on USPTO benchmarks
- Simple HuggingFace interface
- Works on Google Colab without modification
- Supports forward prediction, retrosynthesis, and yield prediction

---

## Installation Commands

### For Google Colab (Recommended)

```python
# Core packages
!pip install rdkit
!pip install torch  # Usually pre-installed
!pip install transformers>=4.40.0
!pip install tokenizers>=0.19.1
!pip install sentencepiece
!pip install accelerate
!pip install datasets
!pip install pandas gdown tqdm plotly

# Optional: Atom mapping
!pip install rxnmapper

# Optional: IBM RXN API
!pip install rxn4chemistry
```

### For Conda Environment

```bash
conda create -n reaction_pred python=3.11 -y
conda activate reaction_pred

# Core packages
conda install -c conda-forge rdkit -y
pip install torch transformers tokenizers sentencepiece
pip install accelerate datasets pandas gdown tqdm plotly

# Optional
pip install rxnmapper
pip install rxn4chemistry
```

---

## Notebook Structure

### Part 1: Setup and Foundations
- Environment detection and package installation
- SMILES representation review
- Tokenization for chemical language models

### Part 2: Reaction Prediction with ReactionT5v2
- Loading pre-trained models from HuggingFace
- Single and batch forward reaction prediction
- Evaluation on USPTO validation set
- Retrosynthesis prediction

### Part 3: Advanced Topics
- Atom mapping with RXNMapper
- IBM RXN for Chemistry API (optional)
- Data augmentation (SMILES randomization)
- Fine-tuning on custom datasets

### Part 4: Visualization and Analysis
- Drawing chemical reactions with RDKit
- Error analysis and debugging
- Common failure patterns

### Appendices
- Legacy OpenNMT-py reference
- Publications and resources

---

## Alternative Approaches Considered

### Option A: Update to OpenNMT-py 3.5.1 (Not Recommended)
- **Status**: Tested and failed
- **Issues**:
  - pyonmttok still not available for Python 3.12
  - Would require downgrading Python, breaking Colab compatibility
  - OpenNMT-py is in maintenance mode

### Option B: HuggingFace Transformers (Implemented)
- **Status**: Recommended approach
- **Advantages**:
  - Modern, actively maintained framework
  - ReactionT5v2 provides state-of-the-art performance
  - Native Colab compatibility
  - Easy fine-tuning

### Option C: IBM RXN API (Supplementary)
- **Status**: Included as optional
- **Advantages**:
  - Production-grade models
  - No local GPU needed
  - Free API access available
- **Limitations**:
  - Requires API key registration
  - Network dependency

### Option D: Eole (Future Consideration)
- **Status**: Not yet mature enough for course use
- **Notes**:
  - OpenNMT-py's successor project
  - May become viable in future course iterations

---

## Data Sources

### USPTO 480K Dataset
- **Description**: ~480,000 reactions from US patents
- **Original Source**: https://figshare.com/articles/dataset/Chemical_reactions_from_US_patents_1976-Sep2016_/5104873
- **Download Links** (in notebook):
  - Training: Google Drive links preserved from original
  - Validation: Used for evaluation
  - Test: Available for final testing

### Open Reaction Database (ORD)
- **Description**: ReactionT5v2's training data
- **Website**: https://open-reaction-database.org/
- **Size**: 2M+ reactions
- **Advantage**: More diverse than patent-only datasets

---

## Performance Expectations

Based on ReactionT5v2 benchmarks:

| Dataset | Top-1 Accuracy | Top-5 Accuracy |
|---------|---------------|----------------|
| USPTO-MIT | ~90% | ~95% |
| USPTO-50K | ~88% | ~93% |

Note: Results may vary based on:
- Reaction complexity
- Presence of stereochemistry
- Whether reaction type was in training data

---

## Known Limitations

1. **Stereochemistry**: USPTO dataset lacks stereochemistry; model may struggle with chiral centers

2. **Rare Reaction Types**: Performance degrades for reactions not well-represented in training

3. **Long Molecules**: Very long SMILES may produce invalid outputs

4. **Yield Prediction**: Less accurate than forward/retro prediction

---

## Files Created

1. **`reaction_prediction_2025.ipynb`**: Updated notebook (42 cells)
   - Path: `/Users/passos/Downloads/studio_12/reaction_prediction_2025.ipynb`

2. **`NOTEBOOK_UPDATE_DOCUMENTATION.md`**: This documentation file
   - Path: `/Users/passos/Downloads/studio_12/NOTEBOOK_UPDATE_DOCUMENTATION.md`

---

## References

### Primary Sources
- [ReactionT5v2 GitHub](https://github.com/sagawatatsuya/ReactionT5v2)
- [ReactionT5v2 HuggingFace](https://huggingface.co/collections/sagawa/reactiont5)
- [RXNMapper GitHub](https://github.com/rxn4chemistry/rxnmapper)
- [rxn4chemistry GitHub](https://github.com/rxn4chemistry/rxn4chemistry)
- [OpenNMT-py Deprecation Notice](https://forum.opennmt.net/t/opennmt-py-is-now-in-maintenance-mode-use-eole-instead/5792)

### Papers
- Sagawa & Kojima, "ReactionT5: a pre-trained transformer model for accurate chemical reaction prediction with limited data", J. Cheminform. (2025)
- Schwaller et al., "Molecular Transformer: A Model for Uncertainty-Calibrated Chemical Reaction Prediction", ACS Cent. Sci. (2019)
- Schwaller et al., "Extraction of organic chemistry grammar from unsupervised learning of chemical reactions", Sci. Adv. (2021)

---

## Contact

For questions about this notebook update, contact the Gomes Group at Carnegie Mellon University.

**Course**: 06-731 Molecular Machine Learning
