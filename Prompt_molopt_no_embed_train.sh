#!/bin/bash
#SBATCH -J No_prompt_v2
#SBATCH -o No_prompt_v2-%j.log
#SBATCH -e No_prompt_v2-%j.err
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH -w gpu21  # If you want to specify a computing node, you can write its name here and remove the first #

LOAD_FROM=""
MODEL=s2s
TASK=no_prompt
MAX_REL_POS=4
ACCUM_COUNT=4
ENC_PE=none
ENC_H=256
BATCH_SIZE=4096
ENC_EMB_SCALE=sqrt
MAX_STEP=500000
ENC_LAYER=4
BATCH_TYPE=tokens
REL_BUCKETS=11

EXP_NO=1
REL_POS=emb_only
ATTN_LAYER=6
LR=4
DROPOUT=0.3


python train_linux_prompt.py \
  --model="$MODEL" \
  --data_name="$DATASET" \
  --task="$TASK" \
  --load_from="$LOAD_FROM" \
  --train_bin="./preprocessed/s2s_mga_data/train_0.npz" \
  --valid_bin="./preprocessed/s2s_mga_data/val_0.npz" \
  --log_file="no_prompt_train.log" \
  --vocab_file="./preprocessed/s2s_mga_data/vocab_smiles.txt" \
  --save_dir="./checkpoints/s2s_no_prompt_data" \
  --embed_size=256 \
  --encoder_num_layers="$ENC_LAYER" \
  --encoder_hidden_size="$ENC_H" \
  --encoder_norm="$ENC_NORM" \
  --encoder_skip_connection="$ENC_SC" \
  --encoder_positional_encoding="$ENC_PE" \
  --encoder_emb_scale="$ENC_EMB_SCALE" \
  --attn_enc_num_layers="$ATTN_LAYER" \
  --attn_enc_hidden_size=256 \
  --attn_enc_heads=8 \
  --attn_enc_filter_size=2048 \
  --rel_pos="$REL_POS" \
  --rel_pos_buckets="$REL_BUCKETS" \
  --decoder_num_layers=6 \
  --decoder_hidden_size=256 \
  --decoder_attn_heads=8 \
  --decoder_filter_size=2048 \
  --dropout="$DROPOUT" \
  --attn_dropout="$DROPOUT" \
  --max_relative_positions="$MAX_REL_POS" \
  --seed=42 \
  --epoch=8000 \
  --max_steps="$MAX_STEP" \
  --warmup_steps=8000 \
  --lr="$LR" \
  --weight_decay=0.0 \
  --clip_norm=20.0 \
  --batch_type="$BATCH_TYPE" \
  --train_batch_size="$BATCH_SIZE" \
  --valid_batch_size="$BATCH_SIZE" \
  --predict_batch_size="$BATCH_SIZE" \
  --accumulation_count="$ACCUM_COUNT" \
  --num_workers=0 \
  --beam_size=5 \
  --n_best=1\
  --predict_min_len=1 \
  --predict_max_len=150 \
  --log_iter=100 \
  --eval_iter=2000 \
  --save_iter=5000
