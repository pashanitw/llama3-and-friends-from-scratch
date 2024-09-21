#!/bin/bash

python finetune.py \
    --max_seq_len 512 \
    --dataset "alpaca_dataset" \
    --model "llama_3" \
    --checkpoint_dir "/path/to/checkpoints" \
    --output_dir "/path/to/output" \
    --batch_size 4 \
    --epochs 5 \
    --learning_rate 1e-5 \
    --device "cuda" \
    --dtype "bf16"