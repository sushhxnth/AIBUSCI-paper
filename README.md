# Architectural Inductive Bias Under Severe Class Imbalance in Exoplanet Detection

**A comparative study of classical machine learning and deep neural architectures using Kepler-derived datasets.**

[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Deep%20Learning-PyTorch-ee4c2c)](https://pytorch.org/)
[![Data](https://img.shields.io/badge/Data-Kepler%20/%20KOI-6f42c1)](https://science.nasa.gov/exoplanets/)

This repository accompanies the research manuscript:

> **Architectural Inductive Bias Under Severe Class Imbalance in Exoplanet Detection: Machine Learning and Deep Neural Architectures on Kepler Light-curves**

The study examines a practical evaluation problem in exoplanet detection: **high overall accuracy can coexist with poor or zero recovery of the rare positive class**. It compares classical classifiers on a structured Kepler Objects of Interest (KOI) dataset with neural architectures evaluated on a highly imbalanced light-curve dataset. The results are interpreted as evidence about the evaluated configurations and data regime—not as a universal ranking of model families.

**Repository:** https://github.com/sushhxnth/AIBUSCI-paper

---

## Contents

- [Research questions](#research-questions)
- [Study design](#study-design)
- [Headline findings](#headline-findings)
- [Datasets](#datasets)
- [Methods](#methods)
- [Evaluation and interpretation](#evaluation-and-interpretation)
- [Reproducibility](#reproducibility)
- [Limitations](#limitations)
- [Citation](#citation)
- [Contact and issues](#contact-and-issues)

## Research questions

The experiments investigate:

1. How do conventional machine-learning classifiers compare on structured KOI features?
2. Under severe class imbalance, do deep models that process light curves identify rare positive examples—or mainly learn to predict the majority class?
3. How do conclusions change when accuracy is considered alongside macro-averaged metrics and confusion matrices?
4. What can the observed outcomes tell us about the evaluated architectures, sampling strategies, and computational constraints?

The emphasis is diagnostic and comparative. This work does **not** introduce a new exoplanet-detection algorithm, and the experiments do not isolate every architectural component through ablation studies.

## Study design

The manuscript evaluates two distinct data settings. They should not be treated as interchangeable benchmarks.

| Setting | Input | Dataset scale | Purpose |
|---|---|---:|---|
| Structured KOI classification | 49 numerical features | 9,564 samples | Compare conventional classifiers |
| Light-curve classification | Time-series flux inputs | 5,087 training/validation-pool examples, plus a separate 570-example test set | Examine neural architectures under severe imbalance |

The structured KOI experiment uses an 80/20 stratified train/test split. The light-curve training/validation pool contains **37 positive examples and 5,050 negative examples** (approximately 135:1). Under the reported split, the training partition contains 30 positives and the validation partition contains 7. The held-out test set contains only **5 positive examples**.

These small positive counts are central to interpreting the results: one additional correctly or incorrectly classified positive example can materially change minority-class metrics.

## Headline findings

### Structured KOI classification

The reported Random Forest test accuracy is **91.17%**, with a macro F1 score of **0.89**. Its confusion matrix is:

```text
[[392,  70,  12],
 [ 70, 396,   5],
 [ 12,   0, 956]]
```

The study also examines feature importance. The leading reported feature is `koi_score`, followed by several KOI disposition/flag and signal-related features. Feature importance is model- and dataset-dependent; it should not be interpreted as causal evidence.

### Light-curve classification

The reported neural experiments show why accuracy alone is inadequate in this setting:

- **CNN–LSTM Hybrid:** 95.97% validation accuracy and 0.70 validation macro recall; on the held-out test set, 98.95% accuracy but **0 of 5 positive examples detected**.
- **CNN without sampler:** 99.31% accuracy, with behavior consistent with majority-class prediction.
- **Transformer without sampler:** 99.31% accuracy, also with majority-class prediction.
- **Transformer with weighted sampling:** 0.69% accuracy and predictions heavily concentrated on the positive class rather than the majority class.

The Transformer configurations therefore failed to discriminate meaningfully between classes in different ways: the sampled configuration predicted nearly all positives, while the unsampled configuration defaulted to the majority class.

**Interpretation:** the strongest headline accuracy is not necessarily the strongest minority-class detector. The held-out test result for the CNN–LSTM Hybrid is a particularly important warning against treating validation performance as sufficient evidence of generalization.

> **Important:** These are the results reported in the manuscript. They should not be read as a stable estimate of rare-event performance: the test set contains only five positive examples, and the study uses a single split.

## Datasets

### 1. Structured KOI dataset

The structured experiment uses a NASA Kepler Objects of Interest (KOI) dataset with 9,564 samples, 49 numerical features, and three target classes.

Reported preprocessing includes median imputation and feature standardization using `StandardScaler`. The experiment uses an 80/20 stratified train/test split.

For the authoritative dataset provenance, feature definitions, target-label mapping, and any filtering decisions, consult the manuscript and the dataset documentation distributed by the original provider. Do not assume that every field in a downloaded KOI table is suitable as a predictive feature: label leakage and features that encode a disposition decision can materially affect interpretation.

### 2. Light-curve dataset

The neural experiments use the Kaggle dataset **“Exoplanet Hunting in Deep Space.”** The reported training/validation pool has 5,087 examples, including 37 positives. A separate test set has 570 examples, including 5 positives.

Most neural models use 3,197 time steps. The Transformer experiment uses 300 time steps because of the computational cost of standard self-attention, which scales quadratically with sequence length.

Obtain the data from its original source and follow that source's terms. The dataset files are not reproduced in this README. Exact file names, expected directory locations, and any preprocessing requirements should be taken from the scripts/notebooks in this repository.

## Methods

### Classical machine learning

The structured-data comparison reports the following test accuracies:

| Model | Reported accuracy |
|---|---:|
| Logistic Regression | 88.71% |
| Decision Tree | 86.10% |
| Decision Tree with maximum-depth constraint | 86.83% |
| Random Forest | 91.17% |
| K-Nearest Neighbors | 81.34% |
| RBF Support Vector Machine | 84.37% |

The Random Forest is also evaluated using its confusion matrix, macro F1, and feature-importance rankings.

### Deep learning

The reported neural configurations include:

- Multilayer Perceptron (MLP)
- Convolutional Neural Network (CNN), with and without weighted sampling
- Long Short-Term Memory network (LSTM)
- Bidirectional LSTM (BiLSTM)
- CNN–LSTM Hybrid
- CNN–LSTM with Attention
- Transformer, with and without weighted sampling

The baseline CNN and CNN–LSTM Hybrid are distinct implementations. The baseline CNN uses two convolutional layers; the hybrid uses three convolutional layers followed by an LSTM. The manuscript reports nine neural experimental configurations across these model variants.

Weighted random sampling is used only in selected configurations. Because architecture, sampling, sequence length, and other training choices are not fully isolated in a factorial or ablation design, differences should be interpreted as comparisons of the **complete evaluated configurations**, not as proof that one component alone caused an outcome.

## Evaluation and interpretation

For imbalanced classification, this study considers accuracy alongside macro-averaged metrics and confusion matrices. These answer different questions:

- **Accuracy** measures the fraction of all examples classified correctly. It can be misleading when one class dominates.
- **Macro recall** gives each class equal weight when averaging class recall, making failures on the minority class more visible.
- **Macro F1** averages class-level F1 scores and penalizes a model that performs poorly on one class.
- **Confusion matrices** reveal which classes are being confused and whether a model predicts only one class.

For rare-positive detection, inspect the positive-class counts and recall directly. In this study, a test set with five positives is too small to support strong claims about expected real-world sensitivity.

## Reproducibility

The repository is intended to accompany the manuscript and its experiments. Before running anything, inspect the scripts/notebooks for their expected data paths, target encoding, preprocessing, random seeds, and configuration settings.

### Environment

The code uses Python scientific-computing and machine-learning libraries, including:

- NumPy
- pandas
- scikit-learn
- Matplotlib
- PyTorch

Install the dependencies and versions specified by the repository's environment/dependency files, if provided. If dependency versions are not pinned, exact numerical reproduction across machines is not guaranteed.

### Run the experiments

1. Clone the repository:

   ```bash
   git clone https://github.com/sushhxnth/AIBUSCI-paper.git
   cd AIBUSCI-paper
   ```

2. Create and activate a Python environment appropriate for your operating system and install the dependencies specified in the repository.

3. Download the datasets from their original sources. Place them in the locations expected by the relevant scripts/notebooks; do not rename columns or change labels unless the code explicitly requires it.

4. Open the relevant notebook or run the corresponding script **using the entry point and arguments documented in the repository**. Check the code before launching a full training run, particularly for data paths, device selection, sequence length, and sampling configuration.

5. Preserve the reported train/validation/test separation. Do not use the held-out test set for model selection or threshold tuning.

6. Compare the resulting confusion matrices and macro metrics—not just accuracy—with the manuscript's reported values.

**Reproduction note:** exact commands and output filenames depend on the actual entry points and file layout in the repository. This README deliberately does not invent script names, command-line flags, pinned package versions, or deterministic-reproduction guarantees. For a fully executable reproduction workflow, those details should be explicitly documented alongside the code.

### Hardware and runtime

The Transformer uses a shorter input sequence (300 time steps) than the other reported light-curve models (typically 3,197 time steps) because of self-attention's quadratic sequence-length cost. Runtime and numerical results can vary with hardware, library versions, random seeds, and accelerator settings. Do not assume that all scripts use a GPU; verify device selection in each implementation.

## Limitations

The results must be interpreted in light of the following constraints:

- **Very few positive examples:** the light-curve training/validation pool contains 37 positives, and the test set contains only five.
- **Single split:** the reported results are not estimates aggregated over repeated splits or cross-validation.
- **No confidence intervals or statistical significance tests:** small differences should not be treated as established effects.
- **No complete ablation study:** the effects of individual architectural components and weighted sampling are not isolated.
- **Different Transformer input length:** its 300-step input differs from the 3,197-step input used by most other neural models.
- **No external validation on an independent mission dataset:** generalization to TESS or other data is not established.
- **Attention maps are descriptive:** visual similarity in attention patterns does not, by itself, establish causal explanations or prove that attention has collapsed to a specific mechanism.

Accordingly, the findings are best understood as a controlled but limited comparison of the reported configurations under severe class imbalance—not as a definitive statement that convolutional or recurrent models are universally superior to Transformers for exoplanet detection.

## Citation

If you use this repository, please cite the associated manuscript. The title and author information below match the manuscript version currently associated with this repository; add the final arXiv identifier or journal reference once it has been assigned.

```text
B. Sushanth Rao,
“Architectural Inductive Bias Under Severe Class Imbalance in Exoplanet
Detection: Machine Learning and Deep Neural Architectures on Kepler
Light-curves.”
```

**Preprint:** Add the arXiv URL/identifier here after the submission is public.

Please also cite the original datasets and relevant methods/software according to their respective citation instructions.

## Contact and issues

For questions about the implementation, reproducibility, or interpretation of the reported experiments, open an issue in this repository and include enough detail to identify the relevant script/notebook, configuration, dataset version, and error or discrepancy.

---

### Scope statement

This repository documents research experiments. It is not a validated operational exoplanet-confirmation system, and its outputs should not be used as standalone evidence that a candidate is a confirmed exoplanet.

## Copyright
© 2026 Sushanth. All Rights Reserved. 

This repository and its contents are provided for academic review and portfolio evaluation purposes only. No permission is granted to use, copy, modify, distribute, or contribute to this code or manuscript.
