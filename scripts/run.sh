# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
python ./src/run.py \
        -train_path=datasets/SFVD1/train.jsonl \
        -val_path=datasets/SFVD1/val.jsonl \
        -test_path=datasets/SFVD1/test_seen.jsonl \
        -val_save_file=datasets/SFVD1/val_results/vision_audio_run39.txt \
        -log_name=PATH_TO_LOG_FILE \
        -batch_size=8 \
        -grad_accum=1 \
        -frame_number=15 \
        -learning_rate=5e-5 \
        -num_epochs=30 \
        -label_shuffle \
        -use_audio \
        -wav2clip \
        -fusion_approach=dot_product \
        -constrained_generation \
        -do_train \
        -n_beams=5 \
        -min_output_len=0 \
        -max_output_len=32 \
        -length_penalty=1 \
        -no_repeat_ngram_size=3
