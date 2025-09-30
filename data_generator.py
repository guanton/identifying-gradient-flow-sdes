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

#!/usr/bin/env python
"""
Module for generating and processing population trajectory data.

This module:
  1. Simulates particle trajectories using an overdamped Langevin SDE.
  2. Processes the trajectories (splitting into training/test, computing couplings, etc.)
  3. Saves the raw and preprocessed data to disk for use in downstream tasks (e.g., JKOnet* training).

It uses the simulate_trajectories helper which now simply wraps
generate_overdamped_langevin_data with a uniform X0 distribution.

Command-line arguments control aspects such as the potential energy, number of timesteps,
dt for the simulation, the number of particles, and seed for reproducibility.
An additional argument, --initial-length (default 4), sets the range for the uniform X0 distribution.
"""

import os
from math import sqrt
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import time
from collections import defaultdict

# Import your simulation and processing functions from the utils submodules.
from utils.langevin_data_generation import generate_overdamped_langevin_data, sample_gibbs, trajectory_overdamped_langevin, sample_5gaussian_mixture, sample_random_gmm
from utils.functions import potentials_all, interactions_all
from utils.rotations   import rotations_all
from utils.sde_simulator import SDESimulator
from utils.density import GaussianMixtureModel
from utils.ot import compute_couplings, compute_couplings_sinkhorn
from utils.plotting import plot_level_curves, create_marginal_gif


# ------------------------------------------------------------------------------
def get_rng(seed, use_jax=False):
    """
    Set the random seed for reproducibility. Returns a jax.random.PRNGKey if use_jax=True,
    otherwise sets and returns the NumPy seed.
    """
    if use_jax:
        return jax.random.PRNGKey(seed)
    else:
        np.random.seed(seed)
        return seed


# ------------------------------------------------------------------------------

def filename_from_args(args):
    """
    Generates a filename based on the provided arguments.
    """
    if args.dataset_name:
        return args.dataset_name

    filename = f"potential_{args.potential}_"
    filename += f"rotation_{args.rotation}_"
    filename += f"internal_{args.internal}_"
    filename += f"beta_{args.beta}_"
    filename += f"interaction_{args.interaction}_"
    filename += f"dt_{args.dt}_"
    filename += f"T_{args.n_timesteps}_"
    filename += f"dim_{args.dimension}_"
    filename += f"N_{args.n_particles}_"
    filename += f"gmm_{args.n_gmm_components}_"
    filename += f"seed_{args.seed}_"
    filename += f"split_{args.test_ratio}"
    filename += f"_split_trajectories_{not args.split_population}"
    filename += f"_lo_{args.leave_one_out}"
    filename += f"_sinkhorn_{args.sinkhorn}"
    return filename


def train_test_split(
        values: jnp.ndarray,
        sample_labels: jnp.ndarray,
        test_ratio: float = 0.0,
        split_trajectories: bool = True,
        seed: int = 0,
):
    """
    Splits the dataset into training and testing subsets while preserving the distribution of labels.
    """
    np.random.seed(seed)
    unique_labels, counts = np.unique(sample_labels, return_counts=True)
    is_balanced = np.all(counts == counts[0])
    assert (not split_trajectories) or is_balanced, "Trajectories are not balanced, cannot split by trajectories."

    if split_trajectories:
        n_particles = counts[0]
        indices = np.arange(n_particles)
        np.random.shuffle(indices)
        test_size = int(n_particles * test_ratio)
        train_indices_block = indices[:-test_size]
        test_indices_block = indices[-test_size:]
        train_indices = []
        test_indices = []
        for label in unique_labels:
            offset = label * n_particles
            train_indices.extend(train_indices_block + offset)
            test_indices.extend(test_indices_block + offset)
    else:
        unique_labels = np.unique(sample_labels)
        train_indices = []
        test_indices = []
        for label in unique_labels:
            idxs = np.where(sample_labels == label)[0]
            np.random.shuffle(idxs)
            split = int(len(idxs) * (1 - test_ratio))
            train_indices.extend(idxs[:split])
            test_indices.extend(idxs[split:])
    return values[np.array(train_indices)], sample_labels[np.array(train_indices)], \
        values[np.array(test_indices)], sample_labels[np.array(test_indices)]


def simulate_trajectories(num_trajectories, d, measurement_times, V, sigma, dt_EM, X0_dist,
                          seed=None, killed = False):
    """
    Simulates trajectories by calling generate_overdamped_langevin_data directly.
    The function X0_dist is expected to take the process dimension d and return a
    d-dimensional vector (i.e., a single initial condition).
    The returned data has shape (n_measurements, num_trajectories, d).
    """
    data = generate_overdamped_langevin_data(
        num_trajectories, d, measurement_times, V, sigma, dt_EM,
        X0_dist=X0_dist, destroyed_samples=False, shuffle=True, seed=seed, killed=killed
    )
    return data.transpose(1, 0, 2)


def generate_data_from_trajectory(
        data_dir: str,
        values: jnp.ndarray,
        sample_labels: jnp.ndarray,
        n_gmm_components: int = 10,
        batch_size: int = 1000,
        leave_one_out: int = -1,
        sinkhorn: float = 0.0
) -> None:
    """
    Process the trajectory data: fit a Gaussian Mixture Model, compute couplings,
    and save density and gradient data to disk.
    """
    sample_labels = [int(label) for label in sample_labels]
    trajectory = defaultdict(list)
    for value, label in zip(values, sample_labels):
        trajectory[label].append(value)
    # Convert each list to a JAX array.
    trajectory = {label: jnp.array(vals) for label, vals in trajectory.items()}
    sorted_labels = sorted(trajectory.keys())
    num_particles_per_step = [trajectory[label].shape[0] for label in sorted_labels]
    is_unbalanced = len(set(num_particles_per_step)) > 1

    if n_gmm_components > 0:
        print("Fitting Gaussian Mixture Model...")
        gmm = GaussianMixtureModel()
        # Note: gmm.fit should accept a seed if desired.
        gmm.fit(trajectory, n_gmm_components, args.seed)

    print("Computing couplings...")
    if sinkhorn > 1e-12:
        f_compute_couplings = lambda x, y, t: compute_couplings_sinkhorn(x, y, t, sinkhorn)
    else:
        f_compute_couplings = lambda x, y, t: compute_couplings(x, y, t)

    for t, label in enumerate(sorted_labels[:-1]):
        if leave_one_out == t or leave_one_out == t + 1:
            continue
        next_label = sorted_labels[t + 1]
        time_t = time.time()
        if is_unbalanced or batch_size < 0:
            couplings = f_compute_couplings(trajectory[label], trajectory[next_label], next_label)
        else:
            couplings = []
            for i in range(int(jnp.ceil(trajectory[0].shape[0] / batch_size))):
                idxs = jnp.arange(i * batch_size, min(trajectory[0].shape[0], (i + 1) * batch_size))
                couplings.append(f_compute_couplings(trajectory[t][idxs, :],
                                                     trajectory[t + 1][idxs, :],
                                                     next_label))
            couplings = jnp.concatenate(couplings, axis=0)
        time_couplings = time.time() - time_t
        print(f"Time to compute couplings: {time_couplings} [s]")
        jnp.save(os.path.join(data_dir, f'couplings_{label}_to_{next_label}.npy'), couplings)

        # Compute densities and gradients.
        ys = couplings[:, (couplings.shape[1] - 1) // 2:-2]
        rho = (lambda _: 0.) if n_gmm_components <= 0 else (lambda x: gmm.gmm_density(t + 1, x))
        densities = jax.vmap(rho)(ys).reshape(-1, 1)
        densities_grads = jax.vmap(jax.grad(rho))(ys)
        data_out = jnp.concatenate([densities, densities_grads], axis=1)
        jnp.save(os.path.join(data_dir, f'density_and_grads_{label}_to_{next_label}.npy'), data_out)


def main(args: argparse.Namespace) -> None:
    """
    Main entry point for generating and processing simulation data.
    """
    print("Running with arguments:", args)

    # ---------------------------------------------------------------
    #  Resolve folder name *first*
    # ---------------------------------------------------------------
    key    = jax.random.PRNGKey(args.seed)
    folder = filename_from_args(args) if args.load_from_file is None else args.load_from_file

    run_dir = os.path.join(args.root, f"p0-{args.p0}", folder)
    data_dir = os.path.join(run_dir, "data")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)


    # --- EARLY-EXIT GUARD -----------------------------------------
    if args.load_from_file is None:
        data_file_path = os.path.join(data_dir, 'data.npy')
        if os.path.isfile(data_file_path):
            print(f"[skip] dataset already exists at '{data_file_path}'.")
            return



    if args.load_from_file is None:
        # --- Define simulation parameters ---
        measurement_times = np.linspace(0, args.n_timesteps * args.dt, args.n_timesteps + 1)

        # Choose potential: use the one from potentials_all if specified; otherwise default to quadratic.
        if args.potential != 'none':
            V = potentials_all[args.potential]
        else:
            V = lambda x: 0.5 * np.sum(x ** 2, axis=-1)

        # Choose divergence-free component
        R = None if args.rotation == 'none' else rotations_all[args.rotation]

        if R is None and args.potential == 'none':
            drift_func = None  # fall back to numerical ∇V==0
        elif R is None:
            drift_func = lambda x: -jax.grad(V)(x)
        elif args.potential == 'none':
            drift_func = R
        else:
            drift_func = lambda x: -jax.grad(V)(x) + R(x)

        # Define sigma (diffusion coefficient) using beta from the Wiener process.
        sigma = sqrt(2 * args.beta)
        dt_EM = min(args.dt_EM, args.dt, 0.01)

        if getattr(args, "five_gaussian", False):
            X0_samples = sample_5gaussian_mixture(
                d=args.dimension,
                total_samples=args.n_particles,
                length=3,
                x0=args.x0,
                seed=args.seed,
                weights=[0.2, 0.2, 0.2, 0.2, 0.2]
            )
            init_name = "5-Gaussian mixture"

        elif getattr(args, "random_gmm", False):
            # Random isotropic GMM in [-gmm-box, gmm-box]^d with σ^2 ∈ [gmm-var-min, gmm-var-max]
            # Returns params so we can log K.
            X0_samples, (means, variances, mix_weights) = sample_random_gmm(
                d=args.dimension,
                total_samples=args.n_particles,
                K=args.gmm_k,  # None ⇒ K~Uniform{1..5}
                box_halfwidth=args.gmm_box,
                var_min=args.gmm_var_min,
                var_max=args.gmm_var_max,
                seed=args.seed,
                return_params=True
            )
            init_name = f"Random GMM (K={means.shape[0]})"

        else:
            X0_samples = sample_gibbs(
                V, sigma, args.dimension,
                total_samples=args.n_particles,
                seed=args.seed,
                T=args.steps_elapsed_X0 * args.dt,
                length=args.initial_length,
                x0=args.x0
            )
            init_name = f"{args.steps_elapsed_X0} steps after Unif"

        print(f"Generating marginals data for the {args.potential} potential function "
              f"where initial distribution is {init_name}.")

        def X0_dist(d):
            # Randomly select one of the pre-sampled initial conditions.
            idx = np.random.randint(0, X0_samples.shape[0])
            return X0_samples[idx]

        print("Generating data...")
        # Use the simulation function (which calls generate_overdamped_langevin_data directly)
        # marginals_data = generate_overdamped_langevin_data(
        #     args.n_particles, args.dimension, measurement_times, V, sigma, dt_EM,
        #     X0_dist=X0_dist, destroyed_samples=False, shuffle=True, seed=args.seed
        # )
        # pass combined drift
        marginals_data = generate_overdamped_langevin_data(args.n_particles, args.dimension, measurement_times, V,
                                                           sigma, dt_EM, X0_dist = X0_dist, destroyed_samples = args.killed,
                                                           shuffle = True, seed = args.seed, drift_func = drift_func)
        if args.rotation == 'none':
            sde_title = f"{args.potential} potential with diffusivity {round(sigma ** 2, 2)}"
        else:
            sde_title = f"{args.potential} potential with {args.rotation} rotation and diffusivity {round(sigma ** 2, 2)}"

        create_marginal_gif(marginals_data, measurement_times, V, sigma, sde_title=sde_title,
                                gif_filename=f'{sde_title}.gif', bins=30, output_dir=run_dir)

        trajectory = marginals_data.transpose(1, 0, 2)
        # Reshape for storage: each row is one sample.
        data = trajectory.reshape(trajectory.shape[0] * trajectory.shape[1], trajectory.shape[2])
        sample_labels = jnp.repeat(jnp.arange(args.n_timesteps + 1), trajectory.shape[1])
        jnp.save(os.path.join(data_dir, 'data.npy'), data)
        jnp.save(os.path.join(data_dir, 'sample_labels.npy'), sample_labels)

        # --- Optional: silently advance to a longer horizon and save ONLY the terminal marginal
        T_record = args.dt * args.n_timesteps
        if args.eq_T is not None and args.eq_T > T_record:
            delta = float(args.eq_T - T_record)
            if delta <= 0:
                print(f"[info] eq_T ({args.eq_T}) <= T_record ({T_record}); skipping extra advance.")
            else:
                # last recorded positions at T_record, shape (N, d)
                X_last = np.asarray(trajectory[-1])  # trajectory: (n_timesteps+1, N, d)
                N, d = X_last.shape
                X_eq = np.empty_like(X_last)

                # Advance each particle from 0 -> delta with the same drift/diffusion (no intermediate storage)
                # IMPORTANT: do NOT pass a seed per call, or you will reset RNG each loop.
                meas2 = [0.0, delta]
                for i in range(N):
                    X_eq[i] = trajectory_overdamped_langevin(
                        meas2, dt_EM, V, sigma, X_last[i], drift_func=drift_func
                    )[-1]
                from pathlib import Path
                out_dir = Path(data_dir)
                np.save(out_dir / "data_eq.npy", X_eq.astype(np.float32))
                with open(out_dir / "eq_meta.txt", "w") as f:
                    f.write(f"T_record={T_record}\n")
                    f.write(f"T_eq={args.eq_T}\n")
                    f.write(f"dt={args.dt}\n")
                    f.write(f"n_extra_steps={int(np.ceil(delta / args.dt))}\n")
                print(f"[info] saved terminal marginal at T={args.eq_T} -> {out_dir / 'data_eq.npy'}")

        # Save run parameters.
        with open(os.path.join(run_dir, 'args.txt'), 'w') as file:
            file.write(f"potential={args.potential}\n")
            file.write(f"internal={args.internal}\n")
            file.write(f"beta={args.beta}\n")
            file.write(f"interaction={args.interaction}\n")
            file.write(f"dt={args.dt}\n")
        if args.potential != 'none':
            potential = potentials_all[args.potential]
            plot_level_curves(potential, ((-4, -4), (4, 4)),
                              save_to=os.path.join(run_dir, 'level_curves_potential'))
    else:
        print("Loading data from file...")
        folder = args.load_from_file
        # recompute run_dir using folder from --load-from-file
        run_dir = os.path.join(args.root, f"p0-{args.p0}", folder)
        data_dir = os.path.join(run_dir, "data")
        data = jnp.load(os.path.join(data_dir, 'data.npy'))
        sample_labels = jnp.load(os.path.join(data_dir, 'sample_labels.npy'))

    # --- Train-test splitting ---
    assert 0 <= args.test_ratio <= 1, "Test split must be a proportion."
    if args.test_ratio > 0:
        train_values, train_labels, test_values, test_labels = train_test_split(
            data, sample_labels, args.test_ratio, not args.split_population, args.seed)
    else:
        train_values, train_labels = data, sample_labels

    jnp.save(os.path.join(data_dir, 'train_data.npy'), train_values)
    jnp.save(os.path.join(data_dir, 'train_sample_labels.npy'), train_labels)
    generate_data_from_trajectory(
        data_dir, train_values, train_labels,
        args.n_gmm_components, args.batch_size,
        args.leave_one_out, args.sinkhorn
    )

    if args.test_ratio > 0:
        jnp.save(os.path.join(data_dir, 'test_data.npy'), test_values)
        jnp.save(os.path.join(data_dir, 'test_sample_labels.npy'), test_labels)

    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-from-file', type=str, default=None,
                        help="Instead of generating data, load a trajectory (shape: (n_timesteps+1, n_particles, dimension)).")
    parser.add_argument(
        '--root', type=str, default='main_experiments',
        help="Root directory where runs are stored (default: main_experiments)."
    )
    parser.add_argument(
        '--p0', type=str, default='gmm', choices=['gmm', 'gibbs', 'unif'],
        help="Initial distribution tag; results saved under p0-<p0>/ (default: gmm)."
    )
    parser.add_argument('--potential', type=str, default='none',
                        choices=list(potentials_all.keys()) + ['none'],
                        help="Name of the potential energy to use.")
    parser.add_argument('--rotation', type=str, default='none', choices = list(rotations_all.keys()) + ['none'],
                        help = "Name of a divergence-free drift term R(x).")
    parser.add_argument("--non_identifiable", action="store_true", help="If set, generate from Gibbs")
    parser.add_argument("--killed", action="store_true", help="If set, samples from marginals are killed")
    parser.add_argument('--n-timesteps', type=int, default=2,
                        help="Number of timesteps of the SDE simulation.")
    parser.add_argument('--dt', type=float, default=0.01,
                        help="Timestep size for the SDE simulation.")
    parser.add_argument('--dt_EM', type=float, default=0.01,
                        help="Timestep size for the SDE simulation.")
    parser.add_argument('--internal', type=str, default='none',
                        choices=['wiener', 'none'],
                        help="Name of the internal energy to use. ('wiener' requires --beta).")
    parser.add_argument('--beta', type=float, default=0.1,
                        help="Standard deviation of the Wiener process (if internal=='wiener').")
    parser.add_argument('--interaction', type=str, default='none',
                        choices=list(interactions_all.keys()) + ['none'],
                        help="Name of the interaction energy to use.")
    parser.add_argument('--dimension', type=int, default=2,
                        help="Dimensionality of the simulated particles.")
    parser.add_argument('--n-particles', type=int, default=2000,
                        help="Number of particles to simulate.")
    parser.add_argument('--batch-size', type=int, default=1000,
                        help="Batch size for computing couplings (negative means no batching).")
    parser.add_argument('--n-gmm-components', type=int, default=10,
                        help="Number of Gaussian Mixture Model components (0 for no GMM).")
    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed for reproducibility.")
    parser.add_argument('--test-ratio', type=float, default=0,
                        help="Proportion of data to allocate to the test set.")
    parser.add_argument('--split-population', action='store_true',
                        help="If set, data is split at every timestep; otherwise by trajectories.")
    parser.add_argument('--leave-one-out', type=int, default=-1,
                        help="Leave one timestep out from training if non-negative.")
    parser.add_argument('--sinkhorn', type=float, default=0.0,
                        help="Regularization parameter for the Sinkhorn algorithm (if < 1e-12, no reg).")
    parser.add_argument('--dataset-name', type=str,
                        help="Optional name for the dataset.")
    parser.add_argument("--steps_elapsed_X0", type=int, default=0)
    parser.add_argument(
        "--x0",
        type=float,
        nargs="+",  # one or more floats
        default=None,  # will mean "origin" if not set
        help="Initial center of X0 distribution. "
             "Pass as space-separated floats, e.g. --x0 4 0 for 2D."
    )
    parser.add_argument("--eq-T", type=float, default=None,
                        help="If set > dt*n_timesteps, continue EM silently to this time and save only the terminal marginal as data_eq.npy")
    # New parameter for the range of initial conditions.
    parser.add_argument('--initial_length', type=float, default=4,
                        help="Half-length for the uniform initial condition, generating X0 in [-initial_length, initial_length].")
    init_group = parser.add_mutually_exclusive_group()
    init_group.add_argument("--five-gaussian", action="store_true",
                            help="Initialize with fixed 5-Gaussian mixture (center+4 corners).")
    init_group.add_argument("--random-gmm", action="store_true",
                            help="Initialize with a random isotropic Gaussian mixture in [-4,4]^d.")

    # --- optional knobs for random-gmm ---
    parser.add_argument("--gmm-k", type=int, default=None,
                        help="Number of components for random GMM; if omitted, K~Uniform{1..10}.")
    parser.add_argument("--gmm-box", type=float, default=3.0,
                        help="Half-width of box for GMM means; means ~ Uniform([-gmm-box,gmm-box]^d).")
    parser.add_argument("--gmm-var-min", type=float, default=0.5,
                        help="Minimum per-component variance σ^2 for random GMM.")
    parser.add_argument("--gmm-var-max", type=float, default=1.0,
                        help="Maximum per-component variance σ^2 for random GMM.")
    args = parser.parse_args()
    main(args)
