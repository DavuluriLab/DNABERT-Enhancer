# DNABERT-Enhancer
## Transformer-based Model for Enhancer Prediction from DNA Sequences
<p align="justify">DNABERT-Enhancer is a deep learning model designed to identify regulatory enhancer elements directly from DNA sequences. The model builds upon the DNABERT architecture and learns contextual representations of genomic sequences to distinguish enhancer regions from non-enhancer regions.
This repository provides the official implementation of DNABERT-Enhancer, including pretrained models, prediction scripts, and evaluation workflows.</p>

## Table of Contents
- [Model Overview](#model-overview)
- [Repository Structure](#repository-structure)
- [Model and Usage](#model-and-usage)
    - [Install Dependencies](#install-dependencies)
    - [Downloading the Pretrained Model](#downloading-the-pretrained-model)
    - [Training the Model](#training-the-model)
    - [Running Prediction](#running-prediction)
    - [Storing Results in W&B](#storing-results-in-W&B)
- [Model and Data Availability](#model-and-data-availability)
- [Citation](#citation)
- [License](#license)

## Model Overview
<p align="justify"> DNABERT-Enhancer is a sequence-based enhancer prediction model that uses transformer-based language modeling for genomic analysis.</p>

Key characteristics of the model:
<ul>
  <li>Transformer-based architecture</li>
  <li>Uses pretrained DNABERT representations</li>
  <li>Fine-tuned on enhancer datasets</li>
  <li>Outputs enhancer probability scores</li>
  <li>Supports genome-scale inference</li>
</ul>

Input:
DNA sequence (fixed-length window)

Output:
Enhancer probability score

## Repository Structure
```
DNABERT-Enhancer/
├── data/                         # Sample data + dataset links
│   ├── raw/                      # Links to original enhancer databases
│   │   └── README.md
│   ├── prediction/               # Information about prediction datasets
│   │   └── README.md
│   └── sample/                   # Small dataset to test the pipeline
│       ├── Train.tsv
│       └── dev.tsv
│
├── src/                          # Source code
│   ├── training/
│   │   ├── Training.sh
│   │   └── run_finetune_WANDB.py
│   └── prediction/
│       └── Predict.sh
│
├── scripts/                      # Additional scripts for processing data
│   ├── Create_input_data.py
│   └── README.md
│
├── models/                       # Fine-tuned models (hosted on Zenodo)
│   └── README.md
│
├── results/                      # Benchmark results and figures
│   └── README.md
│
├── requirements.txt              # Python dependencies
│
└── README.md
```

## Model and Usage
DNABERT-Enhancer is built upon [DNABERT](https://github.com/jerryji1993/DNABERT), a large language model for the human genome, fine-tuned specifically for enhancer prediction. This section describes how to use the code in this repository, including setting up the environment, fine-tuning a new model, and generating predictions with a pre-trained model.

### Install Dependencies
Clone the repository and install the required Python packages. You can install all required packages using the `requirements.txt` file:
```bash
git clone https://github.com/DavuluriLab/DNABERT-Enhancer.git
cd DNABERT-Enhancer
pip install -r requirements.txt
```
Alternatively, using conda:
```bash
conda create -n dnabert_enhancer python=3.10
conda activate dnabert_enhancer
pip install -r requirements.txt
```
**Login to W&B:** The scripts use Weights & Biases for experiment tracking. You will need to log into your W&B account from your terminal.
```bash
wandb login
```

### Downloading the Pretrained Model
---
DNABERT-Enhancer uses the pretrained DNABERT model as the base architecture. The official DNABERT implementation and pretrained weights can be obtained from the DNABERT repository:
https://github.com/jerryji1993/DNABERT
Follow the installation instructions provided in the DNABERT repository.

The Fine-tuned DNABERT-Enhancer models namely, DNABERT-Enhancer-201 and DNABERT-Enhancer-350, built in this study are available on Zenodo:
https://doi.org/10.5281/zenodo.19157566

After downloading, place the model files inside:
```bash
models/
```

### Training the Model
---
The repository includes scripts for fine-tuning DNABERT on enhancer datasets. Open the "Training.sh" script and modify the environment variables at the top to match your system's directory structure. You must set `MODEL_PATH`, `DATA_PATH`, `OUTPUT_PATH`, etc.
Example training command:
```bash
bash src/training/Training.sh
```

### Running Prediction
---
To get predictions on data using the fine-tuned model, use the `Predict.sh` script. Update the `MODEL_PATH` to point to your fine-tuned model directory and `DATA_PATH` to point to the data you want to analyze.
```bash
bash src/prediction/Predict.sh
```

### Storing Results in W&B
---
Both the fine-tuning and prediction scripts are integrated with **Weights & Biases (W&B)** for experiment tracking. When you run the scripts, the following information is automatically logged to your W&B account:

-   **Hyperparameters:** Learning rate, batch size, weight decay, etc.
-   **Performance Metrics:** Training/evaluation loss, accuracy, F1-score, etc.
-   **System Metrics:** GPU/CPU utilization.
-   **Output Files:** Model checkpoints and prediction results can be saved as W&B artifacts.

This allows for easy comparison between runs and ensures full reproducibility of our results. All experiments from our paper are logged and can be viewed in our public W&B project (link to be provided upon publication).

## Model and Data Availability
---
The DNABERT-Enhancer framework is fully open and publicly available.

Code repository:
https://github.com/DavuluriLab/DNABERT-Enhancer

Trained models and associated resources:
https://doi.org/10.5281/zenodo.19157566

Interactive exploration of genome-wide predictions:
https://dnabert-enhancer-datarepo.streamlit.app/

Sample datasets required to test the pipeline are included in this repository.
Links to full datasets and external enhancer databases are provided within the data/ directory.

## Citation
---
If you use the DNABERT-Enhancer in your research, please cite our paper:

```bib

@article{Sathian2025DNABERTEnhancer,
    author = {Sathian, Rekha and Dutta, Pranjal and Ay, Ferhat and Davuluri, Ramana V},
    title = {Genomic Language Model for Predicting Enhancers and Their Allele-Specific Activity in the Human Genome},
    journal = {bioRxiv},
    year = {2025},
    doi = {10.1101/2025.03.18.644040},
    url = {https://doi.org/10.1101/2025.03.18.644040}
```

```bib

@article{ji2021dnabert,
    author = {Ji, Yanrong and Zhou, Zhihan and Liu, Han and Davuluri, Ramana V},
    title = "{DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome}",
    journal = {Bioinformatics},
    volume = {37},
    number = {15},
    pages = {2112-2120},
    year = {2021},
    month = {02},
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btab083},
    url = {https://doi.org/10.1093/bioinformatics/btab083},
    eprint = {https://academic.oup.com/bioinformatics/article-pdf/37/15/2112/50578892/btab083.pdf},
}
```

## License
---
This project is licensed under the Apache License 2.0.
See the LICENSE file for details.
