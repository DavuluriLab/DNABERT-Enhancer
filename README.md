# DNABERT-Enhancer
## Transformer-based Model for Enhancer Prediction from DNA Sequences
<p align="justify">DNABERT-Enhancer is a deep learning model designed to identify regulatory enhancer elements directly from DNA sequences. The model builds upon the DNABERT architecture and learns contextual representations of genomic sequences to distinguish enhancer regions from non-enhancer regions.
This repository provides the official implementation of DNABERT-Enhancer, including pretrained models, prediction scripts, and evaluation workflows.</p>

## Table of Contents
- [Model Overview](#model-overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Using the Pretrained Model](#using-the-pretrained-model)
- [Running Prediction](#running-prediction)
- [Training the Model](#training-the-model)
- [Benchmark Evaluation](#benchmark-evaluation)
- [Genome-wide Application](#genome-wide-application-optional-pipeline)
- [Example Data](#example-data)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [Model and Data Availability](#model-and-data-availability)
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
## Repository Structure
```
DNABERT-Enhancer/
├── data/                         # Sample data + dataset links (full data on Zenodo)
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
├── models/                       # Pretrained models (hosted on Zenodo)
│   └── README.md
│
├── results/                      # Benchmark results and figures
│   └── README.md
│
├── requirements.txt              # Python dependencies
└── README.md
```
## Model and Usage
DNABERT-Enhancer is built upon [DNABERT](https://github.com/jerryji1993/DNABERT), a large language model for the human genome, fine-tuned specifically for enhancer prediction.. This section describes how to use the code in this repository, including setting up the environment, fine-tuning a new model, and generating predictions with a pre-trained model.

  ### Requirements
  Before running the scripts, please install the necessary dependencies. We recommend creating a Python virtual environment. You can install all required packages using the `requirements.txt` file:
  ```bash
  pip install -r requirements.txt
  ```
  **Login to W&B:** The scripts use Weights & Biases for experiment tracking. You will need to log into your W&B account from your terminal.
  ```bash
  wandb login
  ```

### Download Fine-tuned Models
---
The final fine-tuned models used to generate the results in our paper are currently under review. **Upon formal acceptance of the paper, we will upload all model weights to a public repository (e.g., Hugging Face Hub or Zenodo) and provide the download links here.**

### Fine-tuning the Model
---
You can fine-tune the DNABERT model on your own dataset by running the provided training script. This script is configured to perform a hyperparameter search.
1.  **Configure Paths:** Open the `train.sh` script and modify the environment variables at the top to match your system's directory structure. You must set `MODEL_PATH`, `DATA_PATH`, `OUTPUT_PATH`, etc.
2.  **Execute the Script:** Run the script from your terminal.
**`train.sh`**
```bash
#!/bin/bash

# --- 1. CONFIGURE YOUR PATHS ---
export KMER=6
export CLASSES_NAME="Enhancer_NonEnhancer"
export DATA_NAME=24k_true_prediction_data
export DATA_SPLIT=80-20
export ARCHITECTURE="Initial_Screen_Enhancer_Models"
export MODEL_PATH="/path/to/your/pretrained_dnabert/6-new-12w-0" # UPDATE THIS
export DATA_PATH="/path/to/your/data/$DATA_NAME/$DATA_SPLIT"     # UPDATE THIS
export OUTPUT_PATH="/path/to/your/output/Finetuned_models"      # UPDATE THIS
export TB_PATH="/path/to/your/output/TB_Logfiles"               # UPDATE THIS
export SUMMARY_PATH="/path/to/your/output/Results"              # UPDATE THIS
export CUDA_VISIBLE_DEVICES=0,1,2

# --- 2. RUN TRAINING ---
# This script iterates through different learning rates, warmup percentages, and weight decays.
# Ensure you are in the directory containing run_finetune_WANDB.py before running.

learning_rates=("e-3" "e-4" "e-5" "e-6")
BATCH_SIZE=230

# Function to calculate logging and saving steps based on training data size
calculate_steps() {
    local train_file=$1
    local batch_size=$2
    local num_gpus=$(echo "$CUDA_VISIBLE_DEVICES" | tr -cd ',' | wc -c)
    num_gpus=$((num_gpus + 1))
    local num_lines=$(cat "$train_file" | wc -l)
    let "num_examples=num_lines-1" # Exclude header
    let "num_steps=num_examples/(batch_size*num_gpus)"
    let "logging_steps=num_steps*3/2"
    let "save_steps=num_steps*3"
    echo "$logging_steps $save_steps"
}

read logging_steps save_steps <<< $(calculate_steps "$DATA_PATH/train.tsv" $BATCH_SIZE)

for LR in "${learning_rates[@]}"; do
    echo "Processing for Learning Rate: $LR"
    for wp in 0.1 0.2; do
        for wd in 0.0001 0.001 0.005 0.01 0.02; do
            for lr in 1$LR 3$LR; do
                python scripts/run_finetune_WANDB.py \
                    --model_type dna \
                    --tokenizer_name=dna$KMER \
                    --model_name_or_path $MODEL_PATH \
                    --task_name dnaprom \
                    --classes_name $CLASSES_NAME \
                    --architecture $ARCHITECTURE \
                    --do_train \
                    --do_eval \
                    --data_dir $DATA_PATH \
                    --max_seq_length 200 \
                    --per_gpu_eval_batch_size=$BATCH_SIZE   \
                    --per_gpu_train_batch_size=$BATCH_SIZE  \
                    --learning_rate $lr \
                    --num_train_epochs 15 \
                    --output_dir $OUTPUT_PATH \
                    --tb_log_dir $TB_PATH \
                    --summary_dir $SUMMARY_PATH \
                    --evaluate_during_training \
                    --logging_steps $logging_steps \
                    --save_steps $save_steps \
                    --warmup_percent $wp \
                    --hidden_dropout_prob 0.1 \
                    --overwrite_output \
                    --weight_decay $wd \
                    --wandb_tags $CLASSES_NAME $DATA_NAME $LR $DATA_SPLIT \
                    --n_process 36
            done
        done
    done
done
```
### Getting Predictions from the Model
---
To get predictions on new data using a fine-tuned model, use the `predict.sh` script.
1.  **Configure Paths:**  Update the `MODEL_PATH` to point to your fine-tuned model directory and `DATA_PATH` to point to the data you want to analyze.
2.  **Execute the Script:** Run the script from your terminal.
**`predict.sh`**
```bash
#!/bin/bash

# --- 1. CONFIGURE YOUR PATHS ---
export KMER=6
export DATA_NAME=TFBS_prediction
export ARCHITECTURE=TFBS_H3K27ac
export CLASSES_NAME="Enhancer_NonEnhancer"
export MODEL_PATH="/path/to/your/finetuned_model_checkpoint"   # UPDATE THIS
export DATA_PATH="/path/to/your/prediction_data/$DATA_NAME"    # UPDATE THIS
export PREDICTION_PATH="/path/to/your/output/Predictions"      # UPDATE THIS
export SUMMARY_PATH="/path/to/your/output/Results"             # UPDATE THIS
export TB_PATH="/path/to/your/output/TB_Logfiles"              # UPDATE THIS
export CUDA_VISIBLE_DEVICES=1,2,3,4,5

# --- 2. RUN PREDICTION ---
# Ensure you are in the directory containing run_finetune_WANDB.py before running.

python scripts/run_finetune_WANDB.py \
    --model_type dna \
    --tokenizer_name=dna$KMER \
    --model_name_or_path $MODEL_PATH \
    --task_name dnaprom \
    --do_visualize \
    --visualize_data_dir $DATA_PATH \
    --classes_name $CLASSES_NAME \
    --architecture $ARCHITECTURE \
    --visualize_models $KMER \
    --data_dir $DATA_PATH \
    --max_seq_length 200 \
    --per_gpu_pred_batch_size=64 \
    --output_dir $MODEL_PATH \
    --predict_dir $PREDICTION_PATH \
    --tb_log_dir $TB_PATH \
    --wandb_tags $ARCHITECTURE $CLASSES_NAME $DATA_NAME \
    --summary_dir $SUMMARY_PATH \
    --n_process 30
```
### Storing the Results in W&B
---
Both the fine-tuning and prediction scripts are integrated with **Weights & Biases (W&B)** for experiment tracking. When you run the scripts, the following information is automatically logged to your W&B account:

-   **Hyperparameters:** Learning rate, batch size, weight decay, etc.
-   **Performance Metrics:** Training/evaluation loss, accuracy, F1-score, etc.
-   **System Metrics:** GPU/CPU utilization.
-   **Output Files:** Model checkpoints and prediction results can be saved as W&B artifacts.

This allows for easy comparison between runs and ensures full reproducibility of our results. All experiments from our paper are logged and can be viewed in our public W&B project (link to be provided upon publication).

### Model Highlights:
---
<img src="Figures/Model_performance.png" title="Performance metrics of the two DNABERT-Enhancer models">

## Citation
---
If you use the DNABERT-Enhancer in your research, please cite our paper:

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


@misc{zhou2023dnabert2,
      title={DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genome}, 
      author={Zhihan Zhou and Yanrong Ji and Weijian Li and Pratik Dutta and Ramana Davuluri and Han Liu},
      year={2023},
      eprint={2306.15006},
      archivePrefix={arXiv},
      primaryClass={q-bio.GN}
}
