#!/bin/bash
CUDA_LAUNCH_BLOCKING=1 python train.py --training_mode pretrain --data_set pretrain_gloss_data_en_zh --do_train --en --zh --gloss --pretrain_num_epochs 50 --device cuda:0
