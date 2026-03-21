#!/bin/bash
# =========================================================================
# DNABERT-Enhancer prediction script
# Run genome-wide enhancer prediction for a given chromosome
# =========================================================================

# -----------------------------
# Optional environment variables (defaults if not set)
# -----------------------------
export LC_ALL=${LC_ALL:-en_US.utf8}
export LANG=${LANG:-en_US.utf8}
export KMER=${KMER:-6}
export DATA_NAME=${DATA_NAME:-"Whole genome"}
export ARCHITECTURE=${ARCHITECTURE:-"Global model"}
export CLASSES_NAME=${CLASSES_NAME:-"Enhancer_NonEnhancer"}

# -----------------------------
# Repository-relative paths
# -----------------------------
export MODEL_PATH=${MODEL_PATH:-"../../fine_tuned_model"}
export DATA_PATH=${DATA_PATH:-"../../data/$ARCHITECTURE/$DATA_NAME/"}
export PREDICTION_PATH=${PREDICTION_PATH:-"../../results/$ARCHITECTURE/$DATA_NAME/$CHR/Prediction_result"}
export SUMMARY_PATH=${SUMMARY_PATH:-"../../results/$ARCHITECTURE/$DATA_NAME/$CHR/Prediction_result"}
export TB_PATH=${TB_PATH:-"../../logs/$ARCHITECTURE/$DATA_NAME/$CHR/TB_Logfiles"}


# -----------------------------
# Run prediction using fine-tuned DNABERT-Enhancer
# -----------------------------
python src/training/run_finetune_WANDB_rekha.py \
    --model_type dna \
    --tokenizer_name=dna$KMER \
    --model_name_or_path $MODEL_PATH \
    --architecture $ARCHITECTURE \
    --task_name dnaprom \
    --classes_name $CLASSES_NAME \
    --do_predict \
    --data_dir $DATA_PATH \
    --max_seq_length 350 \
    --per_gpu_pred_batch_size=100 \
    --output_dir $MODEL_PATH \
    --summary_dir $SUMMARY_PATH \
    --predict_dir $PREDICTION_PATH \
    --tb_log_dir $TB_PATH \
    --wandb_tags $ARCHITECTURE $DATA_NAME \
    --n_process 36

echo "DNABERT-Enhancer prediction completed."