# Genomic Language Model for Predicting Enhancers and Their Allele-Specific Activity in the Human Genome

## Overview
<p align="justify"> DNABERT-Enhancer is a machine learning method based on the genome foundational large language model DNABERT. The fine-tuned model successfully identify enhancer regions in the human genome by focusing on the high-attention regions in the enhancer sequences. Moreover, DNABERT-Enhancer is applied to identify and predict candidate non-coding genomic variants in the enhancer regions which can cause a significant shift in enhancer function as well as disrupt the  TFBSs within. This repository hosts the DNABERT-Enhancer model, datasets, and related resources. </p>

## Table of Contents
- [Introduction](#introduction)
- [Abstract](#abstract)
- [Model and Usage](#model)
  - [Usage](#usage)
    - [Download Fine-tuned Model](#download-fine-tuned-model)
    - [Getting Prediction from the Model](#getting-prediction-from-the-model)
    - [Storing the Results in W&B](#storing-the-results-in-wb)
- [Citation](#citation)
- [Contact](#contact)

## Introduction
<p align="justify"> Enhancers coordinate the gene expression in a tissue-specific manner. Computational identification of these non-coding regulatory regions is challenging because of its intricate characteristics like varying lengths, ambiguous boundaries, functioning independent of their orientation and variable distances from associated promoters. Moreover, absence of well-defined genetic or epigenetic signatures that can accurately discriminate enhancers within genomic regions adds another layer of complexity. Enhancer malfunction due to genetic alteration mechanisms cause aberrant gene expression leading to genetic diseases like congenital disorders, cancers, and common complex diseases, collectively termed as enhanceropathies. Identifying disease-associated enhancers and recognizing GWAS-identified risk variants within them opens window for drug target discovery and enhancer-targeting therapies implying the clinical significance. Traditional machine learning methods often struggle to grasp the true context and the nuanced nature of enhancer DNA sequences, leading to uncertain real-world efficacy for enhancer prediction. </p>

## Abstract
<p align="justify">Predicting and deciphering the regulatory logic of enhancers is a challenging problem, due to the intricate sequence features and lack of consistent genetic or epigenetic signatures that can accurately discriminate enhancers from other genomic regions. Recent machine-learning based methods have spotlighted the importance of extracting nucleotide composition of enhancers, but models based solely on them fail to learn the sequence context and perform suboptimally.  Motivated by advances in genomic language models, we developed a novel enhancer prediction method, called DNABERT-Enhancer, by applying DNABERT pre-trained language model on the human genome. We trained two different models, one on 17,540 enhancers and the other on 36,927 enhancers curated from the ENCODE registry of candidate cis-Regulatory Elements (cCREs). The best fine-tuned model achieved performance of 88.05% accuracy with Matthews correlation coefficient (MCC) of 76% on independent set aside data. Further, we present the analysis of the predicted enhancers for all chromosomes of the human genome by comparing with the enhancer regions reported in publicly available databases. Finally, we applied DNABERT-Enhancer along with other DNABERT fine-tuned models to predict candidate SNPs with allele-specific enhancer and TF binding activity. The genome-wide enhancer annotations and candidate loss-of-function genetic variants predicted by DNABERT-Enhancer provide valuable resources for genome interpretation in functional and clinical genomics studies. </p>
<img src="Figures/Graphical_abstract.png" title="Graphical abstract">

## Model
<p align="justify"> DNABERT-Enhancer is built upon [DNABERT](https://github.com/jerryji1993/DNABERT), a large language model for the human genome, fine-tuned specifically for enhancer prediction. </p>

## Usage of the Model
Instructions on how to effectively use the DNABERT-Enhancer model.

  - #### **Download Fine-tuned Model and Prediction :** Details on downloading the model and instructions for fine-tuning and prediction will be provided upon request. 
  Requirements
Before running the scripts, please install the necessary dependencies. We recommend creating a Python virtual environment. You can install all required packages using the requirements.txt file:

Bash

pip install -r requirements.txt
(You should create a requirements.txt file containing libraries like torch, transformers, wandb, scikit-learn, etc.)

You will also need to be logged into your Weights & Biases account to log experiment results:

Bash

wandb login
Download Fine-tuned Models
The final fine-tuned models used to generate the results in our paper are currently under review. Upon formal acceptance of the paper, we will upload all model weights to a public repository (e.g., Hugging Face Hub or Zenodo) and provide the download links here.

Fine-tuning the Model
You can fine-tune the DNABERT model on your own dataset by running the provided training script. This script performs a hyperparameter search to optimize the model.

Configure Paths: Open the script and modify the environment variables at the top to match your system's directory structure. You must set MODEL_PATH, DATA_PATH, OUTPUT_PATH, etc.

Execute the Script: Run the following command from the repository's root directory.

Bash

#!/bin/bash

# --- Configuration ---
export KMER=6
export CLASSES_NAME="Enhancer_NonEnhancer"
export DATA_NAME=24k_true_prediction_data
export DATA_SPLIT=80-20
export ARCHITECTURE="Initial_Screen_Enhancer_Models"
export MODEL_PATH="/path/to/your/pretrained_model/6-new-12w-0" # UPDATE THIS
export DATA_PATH="/path/to/your/data/$DATA_NAME/$DATA_SPLIT"  # UPDATE THIS
export OUTPUT_PATH="/path/to/your/output/Finetuned_models"   # UPDATE THIS
export TB_PATH="/path/to/your/output/TB_Logfiles"            # UPDATE THIS
export SUMMARY_PATH="/path/to/your/output/Results"           # UPDATE THIS
export CUDA_VISIBLE_DEVICES=0,1,2

# --- Training Script ---
# This script iterates through different learning rates, warmup percentages, and weight decays.
learning_rates=("e-3" "e-4" "e-5" "e-6")
BATCH_SIZE=230

for LR in "${learning_rates[@]}"; do
    echo "Processing for Learning Rate: $LR"
    for wp in 0.1 0.2; do
        for wd in 0.0001 0.001 0.005 0.01 0.02; do
            for lr in 1$LR 3$LR; do
                python run_finetune_WANDB.py \
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
                    --per_gpu_train_batch_size=$BATCH_SIZE \
                    --learning_rate $lr \
                    --num_train_epochs 15 \
                    --output_dir $OUTPUT_PATH \
                    --warmup_percent $wp \
                    --weight_decay $wd \
                    --overwrite_output \
                    --wandb_tags $CLASSES_NAME $DATA_NAME $LR $DATA_SPLIT
            done
        done
    done
done
Getting Predictions from the Model
To get predictions on new data using a fine-tuned model, use the prediction script below.

Configure Paths: Update the MODEL_PATH to point to your fine-tuned model directory and DATA_PATH to point to the data you want to analyze.

Execute the Script:

Bash

#!/bin/bash

# --- Configuration ---
export KMER=6
export DATA_NAME=TFBS_prediction
export ARCHITECTURE=TFBS_H3K27ac
export CLASSES_NAME="Enhancer_NonEnhancer"
export MODEL_PATH="/path/to/your/finetuned_model_checkpoint"   # UPDATE THIS
export DATA_PATH="/path/to/your/prediction_data/$DATA_NAME" # UPDATE THIS
export PREDICTION_PATH="/path/to/your/output/Predictions"     # UPDATE THIS
export CUDA_VISIBLE_DEVICES=1,2,3,4,5

# --- Prediction Script ---
python run_finetune_WANDB.py \
    --model_type dna \
    --tokenizer_name=dna$KMER \
    --model_name_or_path $MODEL_PATH \
    --task_name dnaprom \
    --do_visualize \
    --visualize_data_dir $DATA_PATH \
    --classes_name $CLASSES_NAME \
    --architecture $ARCHITECTURE \
    --max_seq_length 200 \
    --per_gpu_pred_batch_size=64 \
    --output_dir $MODEL_PATH \
    --predict_dir $PREDICTION_PATH \
    --wandb_tags $ARCHITECTURE $CLASSES_NAME $DATA_NAME
Storing Results in W&B
Both the fine-tuning and prediction scripts are integrated with Weights & Biases (W&B) for experiment tracking. When you run the scripts, the following information is automatically logged to your W&B account:

Hyperparameters: Learning rate, batch size, weight decay, etc.

Performance Metrics: Training/evaluation loss, accuracy, F1-score, etc.

System Metrics: GPU/CPU utilization.

Output Files: Model checkpoints and prediction results can be saved as W&B artifacts.

This allows for easy comparison between runs and ensures full reproducibility of our results. All experiments from our paper are logged and can be viewed in our public W&B project (link to be provided upon publication).

  
  - #### Model Highlights:
    <img src="Figures/Model_performance.png" title="Performance metrics of the two DNABERT-Enhancer models">

## Citation
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
