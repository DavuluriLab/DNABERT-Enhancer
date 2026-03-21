#!/bin/bash
# =========================================================================
# DNABERT-Enhancer training script
# Fine-tunes DNABERT for enhancer prediction
# =========================================================================

# -----------------------------
# Optional environment variables (use defaults if not set)
# -----------------------------
export LC_ALL=${LC_ALL:-en_US.utf8}
export LANG=${LANG:-en_US.utf8}
export KMER=${KMER:-6}
export CLASSES_NAME=${CLASSES_NAME:-"Enhancer_Non-enhancer"}
export TAG_NAME=${TAG_NAME:-"350bp"}
export DATA_NAME=${DATA_NAME:-"SCREEN"}
export DATA_SPLIT=${DATA_SPLIT:-"80-20"}
export ARCHITECTURE=${ARCHITECTURE:-"Global_model"}
export NUM_EPOCHS=${NUM_EPOCHS:-20}
export BATCH_SIZE=${BATCH_SIZE:-400}

# -----------------------------
# Repository-relative paths
# -----------------------------
export MODEL_PATH=${MODEL_PATH:-"../../models/pretrained_dnabert"}
export DATA_PATH=${DATA_PATH:-"../../data/$ARCHITECTURE/$DATA_NAME/$DATA_SPLIT"}
export OUTPUT_PATH=${OUTPUT_PATH:-"../../models/$ARCHITECTURE/$DATA_NAME/$DATA_SPLIT/Finetuned_models"}
export TB_PATH=${TB_PATH:-"../../logs/$ARCHITECTURE/$DATA_NAME/$DATA_SPLIT/TB_Logfiles"}
export SUMMARY_PATH=${SUMMARY_PATH:-"../../results/$ARCHITECTURE/$DATA_NAME/$DATA_SPLIT/Results"}

# Move to DNABERT examples folder where run_finetune_WANDB.py
cd ..

# -----------------------------
# Function to calculate logging and saving steps
# -----------------------------
calculate_steps() {
    local train_file=$1
    local batch_size=$2
    local num_lines
    num_lines=$(wc -l < "$train_file")
    local num_examples=$((num_lines-1)) # exclude header
    local num_steps=$((num_examples / (batch_size*2)))
    local logging_steps=$((num_steps*3/2))  # Log ~10 times per epoch
    local save_steps=$((num_steps*2))       # Save ~2 times per epoch
    echo "$logging_steps $save_steps"
}

# -----------------------------
# Learning rates, weight decay, warmup percentages
# -----------------------------
learning_rates=("1e-3" "3e-3" "1e-4" "3e-4" "1e-5" "3e-5" "1e-6" "3e-6")
weight_decays=(0.0001 0.001 0.005 0.01 0.02)
warmup_percents=(0.1 0.2)

# -----------------------------
# Calculate logging and saving steps
# -----------------------------
read logging_steps save_steps <<< $(calculate_steps "$DATA_PATH/train.tsv" $BATCH_SIZE)

# -----------------------------
# Training loop
# -----------------------------
for LR in "${learning_rates[@]}"; do
    echo "Processing for Learning Rate: $LR"
    for wp in "${warmup_percents[@]}"; do
        for wd in "${weight_decays[@]}"; do
            python src/training/run_finetune_WANDB.py \
                --model_type dna \
                --tokenizer_name=dna$KMER \
                --model_name_or_path $MODEL_PATH \
                --task_name dnaprom \
                --classes_name $CLASSES_NAME \
                --architecture $ARCHITECTURE \
                --do_train \
                --do_eval \
                --data_dir $DATA_PATH \
                --max_seq_length 350 \
                --per_gpu_train_batch_size=$BATCH_SIZE \
                --per_gpu_eval_batch_size=$BATCH_SIZE \
                --learning_rate $LR \
                --num_train_epochs $NUM_EPOCHS \
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
                --wandb_tags $ARCHITECTURE $CLASSES_NAME $DATA_NAME $TAG_NAME $LR $DATA_SPLIT \
                --n_process 36
        done
    done
done

echo "DNABERT-Enhancer training completed."