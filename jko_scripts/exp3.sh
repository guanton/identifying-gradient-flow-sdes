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
potentials=("wavy_plateau" "double_exp" "rotational" "relu" "flat" "friedman" "watershed" "ishigami" "flowers" "bohachevsky" "sphere" "styblinski_tang" "oakley_ohagan" "zigzag_ridge" "holder_table")
interactions=("wavy_plateau" "double_exp" "rotational" "relu" "flat" "friedman" "watershed" "ishigami" "flowers" "bohachevsky" "sphere" "styblinski_tang" "oakley_ohagan" "zigzag_ridge")
betas=(0.0 0.1 0.2)

export potentials interactions betas

parallel -j 8 "
    potential={1}; interaction={2}; beta={3};
    python data_generator.py --potential \$potential --interaction \$interaction --n-particles 2000 --test-ratio 0.5 --internal wiener --beta \$beta &&
    python train.py --solver jkonet-star --dataset potential_\${potential}_internal_wiener_beta_\${beta}_interaction_\${interaction}_dt_0.01_T_5_dim_2_N_2000_gmm_10_seed_0_split_0.5_split_trajectories_True_lo_-1_sinkhorn_0.0 --wandb &&
    python train.py --solver jkonet-star-linear --dataset potential_\${potential}_internal_wiener_beta_\${beta}_interaction_\${interaction}_dt_0.01_T_5_dim_2_N_2000_gmm_10_seed_0_split_0.5_split_trajectories_True_lo_-1_sinkhorn_0.0 --wandb
" ::: "${potentials[@]}" ::: "${interactions[@]}" ::: "${betas[@]}"