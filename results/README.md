## Model performance
DNABERT-Enhancer includes two models namely, DNABERT-Enhancer-201 and DNABERT-Enhancer-350, trained on enhancer data of 201 bp and 350 bp respectively. The models were compared to the Base-line methods, Recent enhancer prediction methods and the Nucleotide traansformer.

### Comparison to Base-line methods
| Models | Eds-201-Test Accuracy (%) | Precision (%) | Recall (%) | F1 score (%) | MCC (%) | Eds-350-Test Accuracy (%) | Precision (%) | Recall (%) | F1 score (%) | MCC (%) |
|--------|---------------------------|--------------|------------|--------------|---------|---------------------------|--------------|------------|--------------|---------|
| Random Forest | 74.74 | 75.36 | 73.51 | 74.42 | 49.49 | 78.12 | 79.49 | 75.81 | 77.61 | 56.31 |
| KNN | 57.1 | 82.01 | 18.19 | 29.78 | 22.62 | 53.63 | 95.4 | 7.64 | 14.14 | 18.54 |
| SVC | 73.47 | 73.56 | 73.28 | 73.42 | 46.95 | 78.07 | 77.47 | 79.15 | 78.3 | 56.14 |
| Gaussian Naive Bayes | 69.56 | 71.2 | 65.71 | 68.34 | 39.24 | 72.58 | 74.42 | 68.83 | 71.51 | 45.3 |
| AdaBoost Classifier | 75.73 | 75.4 | 76.38 | 75.89 | 51.46 | 77.9 | 77.8 | 78.07 | 77.93 | 55.8 |
| MLP | 69.24 | 68.99 | 69.9 | 69.45 | 38.49 | 77.2 | 76.84 | 77.88 | 77.36 | 54.41 |
| DNABERT-Enhancer | **82.04** | **84.64** | **78.29** | **81.3** | **64.27** | **88.05** | **90.27** | **85.29** | **87.71** | **76.22** |
