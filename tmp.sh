#!/bin/bash 

python3 -B LDA_Gibbs.py --n_train 700 --run_train --n_samples   5
python3 -B LDA_Gibbs.py --n_train 700 --run_train --n_samples  50
