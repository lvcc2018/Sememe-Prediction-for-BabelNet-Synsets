#!/bin/bash
CUDA_LAUNCH_BLOCKING=1 python train.py --training_mode train --do_eval --data_set all_data_a --load_model output/xlm-roberta-base_all_data_word_en_zh_fr.pt --en --fr --zh --word
CUDA_LAUNCH_BLOCKING=1 python train.py --training_mode train --do_eval --data_set all_data_v --load_model output/xlm-roberta-base_all_data_word_en_zh_fr.pt --en --fr --zh --word
CUDA_LAUNCH_BLOCKING=1 python train.py --training_mode train --do_eval --data_set all_data_n --load_model output/xlm-roberta-base_all_data_word_en_zh_fr.pt --en --fr --zh --word
CUDA_LAUNCH_BLOCKING=1 python train.py --training_mode train --do_eval --data_set all_data_r --load_model output/xlm-roberta-base_all_data_word_en_zh_fr.pt --en --fr --zh --word