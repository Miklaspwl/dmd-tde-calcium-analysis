# DMD-TDE analysis of astrocytic Ca2+ signaling

This repository contains example code and datasets for reproducing the delay-embedded dynamic mode decomposition (DMD-TDE) analysis used in:

"Single-cell analysis of sterol-induced Ca2+ signaling in human astrocytes by dynamic mode decomposition"

Contents:

* Example normalized Ca2+ datasets (biological replicate 3)
* Jupyter notebook implementing the DMD-TDE workflow
* Clustering analysis using kernel PCA and k-means
* Example output generation

## Files

* `example_dmd_tde.ipynb`
  Main analysis notebook

* `*_003_raw_signals_normalized.csv`
  Example normalized Ca2+ traces for replicate 3

## Dependencies

```text
numpy
pandas
matplotlib
scikit-learn
```

## Usage

Open the notebook and run all cells sequentially to reproduce the example analysis workflow.
