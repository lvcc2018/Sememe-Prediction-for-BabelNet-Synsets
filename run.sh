#!/bin/bash
python train.py --do_train --gloss --en --fr --zh --num_epochs 40
python train.py --do_train --word --en --fr --zh --num_epochs 40
