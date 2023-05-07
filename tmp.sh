#!/bin/bash 


python3 -B LDA_Gibbs.py --n_train 50  --run_train
python3 -B LDA_Gibbs.py --n_train 100 --run_train
python3 -B LDA_Gibbs.py --n_train 200 --run_train
python3 -B LDA_Gibbs.py --n_train 400 --run_train
python3 -B LDA_Gibbs.py --n_train 600 --run_train

