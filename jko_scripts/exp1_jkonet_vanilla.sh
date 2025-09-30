# MIT License
#
# Copyright (c) 2024 Antonio Terpin, Nicolas Lanzetti, Martin Gadea, Florian Dörfler
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#!/bin/bash
potentials=("zigzag_ridge" "double_exp" "rotational" "relu" "flat" "friedman" "watershed" "ishigami" "flowers" "bohachevsky" "holder_table" "wavy_plateau" "oakley_ohagan" "sphere" "styblinski_tang")

parallel -j 8 "
    python data_generator.py --potential {} --n-particles 2000 --test-ratio 0.5 &&
    python train.py --solver jkonet-vanilla --dataset potential_{}_internal_none_beta_0.0_interaction_none_dt_0.01_T_5_dim_2_N_2000_gmm_10_seed_0_split_0.5_split_trajectories_True_lo_-1_sinkhorn_0.0 --wandb
" ::: "${potentials[@]}"