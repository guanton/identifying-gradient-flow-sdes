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

potentials=("wavy_plateau" "oakley_ohagan" "double_exp" "rotational" "relu" "flat" "friedman" "watershed" "ishigami" "flowers" "bohachevsky" "sphere" "styblinski_tang" "zigzag_ridge" "holder_table")
dims=(10 20 30 40 50)
n_particles_values=(2000 5000 10000 15000 20000)

export potentials dims n_particles_values

parallel -j 8 "
    potential={1}; dim={2}; n_particles={3};
    python data_generator.py --potential \$potential --n-particles \$n_particles --test-ratio 0.5 --dimension \$dim &&
    python train.py --solver jkonet-star-potential --dataset potential_\${potential}_internal_none_beta_0.0_interaction_none_dt_0.01_T_5_dim_\${dim}_N_\${n_particles}_gmm_10_seed_0_split_0.5_split_trajectories_True_lo_-1_sinkhorn_0.0 --wandb
" ::: "${potentials[@]}" ::: "${dims[@]}" ::: "${n_particles_values[@]}"
