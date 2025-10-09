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

"""
This module provides a script for training and evaluating JKOnet* and other models for learning diffusion terms on population data.

Functions
-------------
- ``numpy_collate``
    A custom collate function for PyTorch's DataLoader to properly stack or nest NumPy arrays when using JAX.

- ``main``
    The main function that orchestrates the training loop, evaluation, logging, and visualization. It reads configurations, initializes models and datasets, and executes the training and evaluation processes.

Command-Line arguments
----------------------
The script accepts the following command-line arguments:

- `--solver`, `-s` (`EnumMethod`):
    Name of the solver (model) to use. Choices are defined in the `EnumMethod` class.

- `--dataset`, `-d` (`str`):
    Name of the dataset to train the model on. The dataset should be prepared and located in a directory matching this name.

- `--eval` (`str`):
    Option to test the fit on `'train_data'` or `'test_data'` (e.g., for debugging purposes). Default is `'test_data'`.

- `--wandb` (`bool`):
    If specified, activates Weights & Biases logging for experiment tracking.

- `--debug` (`bool`):
    If specified, runs the script in debug mode (disables JIT compilation in JAX for easier debugging).

- `--seed` (`int`):
    Seed for random number generation to ensure reproducibility.

- `--epochs` (`int`):
    Number of epochs to train the model. If not specified, the number of epochs is taken from the configuration file.

Usage example
-------------
To train a model using the `jkonet-star-potential` solver on a dataset named `my_dataset` with wandb logging:

.. code-block:: bash

    python train.py --solver jkonet-star-potential --dataset my_dataset --wandb

"""

import os
import json
import jax
import yaml
import torch
import wandb
import argparse
from time import time
from tqdm import tqdm
import numpy as np
import jax.numpy as jnp
from torch.utils.data import DataLoader
from models import EnumMethod, get_model
from models.nn_APPEX import sb_refine
from dataset import PopulationEvalDataset
from utils.sde_simulator import get_SDE_predictions
from typing import Union, List, Tuple
from jko_scripts.load_from_wandb import wandb_config
from utils.functions import potentials_all
from utils.rotations import rotations_all
from utils.langevin_data_generation import numerical_gradient

# --- helpers to export an evaluation-ready drift & a reusable bundle -----

def _wrap_numpy_fn(fn):
    """Wrap a JAX function so it accepts/returns NumPy on CPU cleanly."""
    def g(x_np: np.ndarray) -> np.ndarray:
        x_j = jnp.asarray(x_np)
        y_j = fn(x_j)
        return np.asarray(y_j)
    return g

def _get_estimated_drift_fn(model, state, use_refined, refined_drift_fn):
    """
    Return a NumPy-friendly callable est_drift_fn(x_np:[d])->[d] for evaluation/saving.
    - If `use_refined` is True, use `refined_drift_fn`.
    - Else, compute -∇(model.get_potential(state)).
    """
    if use_refined and refined_drift_fn is not None:
        # refined_drift_fn already maps ndarray->ndarray in NumPy/JAX; just wrap to be safe
        return _wrap_numpy_fn(lambda x: refined_drift_fn(x))
    else:
        pot = model.get_potential(state)  # JAX scalar potential: R^d -> R
        # JITing the gradient gives a nice speedup for grid export
        drift_jax = jax.jit(lambda x: -jax.grad(pot)(x))
        return _wrap_numpy_fn(drift_jax)

def _save_inference_bundle(
    out_dir: str,
    *,
    dim: int,
    grid_size: float,
    resolution: int,
    est_sigma2: float,          # σ²
    est_drift_fn,               # callable: np.ndarray[d] -> np.ndarray[d]
    seed: int = 0,
    n_random: int = 100,
    init_sigma2: float = 1.0
):
    """
    Save a reusable bundle so you can compute arbitrary metrics later
    without retraining:
      - a uniform grid ([-grid_size, grid_size]^d) and the drift on that grid
      - a large set of random points in the same box and their drift
      - metadata including σ²

    Files written:
      out_dir/inference_bundle.npz
      out_dir/inference_meta.json
    """
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.RandomState(seed)

    # ----- grid points -----
    if dim == 1:
        xs = np.linspace(-grid_size, grid_size, resolution)
        Xg = xs[:, None]                            # [M,1]
    elif dim == 2:
        xs = np.linspace(-grid_size, grid_size, resolution)
        ys = np.linspace(-grid_size, grid_size, resolution)
        X, Y = np.meshgrid(xs, ys, indexing="xy")
        Xg = np.stack([X.ravel(), Y.ravel()], axis=1)  # [M,2]
    else:
        M = resolution ** 2                          # keep size reasonable
        Xg = rng.uniform(-grid_size, grid_size, size=(M, dim))

    # evaluate drift on grid
    Gg = np.stack([est_drift_fn(x) for x in Xg], axis=0).astype(np.float32)  # [M,d]

    # ----- random cloud -----
    Xr = rng.uniform(-grid_size, grid_size, size=(n_random, dim)).astype(np.float32)
    Gr = np.stack([est_drift_fn(x) for x in Xr], axis=0).astype(np.float32)  # [n_random,d]

    # save arrays
    np.savez_compressed(
        os.path.join(out_dir, "inference_bundle.npz"),
        grid_points=Xg.astype(np.float32),
        grid_drift=Gg,
        random_points=Xr,
        random_drift=Gr,
    )

    # save meta
    meta = {
        "init_sigma_2": float(init_sigma2),
        "dim": int(dim),
        "grid_size": float(grid_size),
        "resolution": int(resolution),
        "sigma2": float(est_sigma2),
        "seed": int(seed),
    }
    with open(os.path.join(out_dir, "inference_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

def _safe_scalar(x, default=0.1, lo=1e-6, hi=1000000.0):
    """Return a finite float in [lo, hi]; fall back to default if NaN/Inf/out-of-range."""
    xv = float(np.asarray(x))
    if not np.isfinite(xv) or xv <= 0:
        print(f"[guard] non-finite/invalid scalar {xv}; using default {default}")
        return float(default)
    return float(np.clip(xv, lo, hi))

def _all_finite(arr):
    a = np.asarray(arr)
    return np.isfinite(a).all()



def numpy_collate(batch: List[Union[np.ndarray, Tuple, List]]) -> Union[np.ndarray, List]:
    """
    Collates a batch of samples into a single array or nested list of arrays.

    This function recursively processes a batch of samples, stacking NumPy arrays, and collating lists or tuples by grouping elements together. If the batch consists of NumPy arrays, they are stacked. If the batch contains tuples or lists, the function recursively applies the collation.

    This collate function is taken from the `JAX tutorial with PyTorch Data Loading <https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html>`_.

    Parameters
    ----------
    batch : List[Union[np.ndarray, Tuple, List]]
        A batch of samples where each sample is either a NumPy array, a tuple, or a list. It depends on the
        data loader.

    Returns
    -------
    np.ndarray
        The collated batch, either as a stacked NumPy array or as a nested structure of arrays.
    """
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

def _make_true_drift_fn(potential_name: str, rotation_name: str):
    """Return b_true(X): (N,d)->(N,d) implementing -∇V(x) (+ rotation if any)."""
    true_V = (potentials_all[potential_name]
              if potential_name != "none"
              else (lambda x: 0.5 * np.sum(x ** 2)))
    R_fn = None if rotation_name == "none" else rotations_all[rotation_name]

    def b_true(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            g = -numerical_gradient(true_V, X)
            if R_fn is not None: g = g + np.asarray(R_fn(X))
            return np.asarray(g)
        # batch
        grads = np.stack([-numerical_gradient(true_V, x) for x in X], axis=0)
        if R_fn is not None:
            rots = np.stack([np.asarray(R_fn(x)) for x in X], axis=0)
            grads = grads + rots
        return grads
    return b_true

def compute_drift_metrics(
    true_V, est_V, d,
    grid_size=5.0, resolution=50,
    normalize=True
):
    """Return grid_mae, grid_cos, gibbs_mae, gibbs_cos (last two = None)."""

    # ---------- grid points ------------------------------------------
    if d == 1:
        xs = np.linspace(-grid_size, grid_size, resolution)
        points = [np.array([x]) for x in xs]
    elif d == 2:
        xs = np.linspace(-grid_size, grid_size, resolution)
        ys = np.linspace(-grid_size, grid_size, resolution)
        X, Y = np.meshgrid(xs, ys)
        points = [np.array([x, y]) for x, y in zip(X.ravel(), Y.ravel())]
    else:
        points = np.random.uniform(-grid_size, grid_size, size=(1000, d))

    # ---------- evaluation ------------------------------------------
    abs_err, ref_mag, cos_vals = [], [], []
    for x in points:
        t = np.ravel(-numerical_gradient(true_V, x)).astype(float)   # (d,)
        e = np.ravel(-numerical_gradient(est_V,  x)).astype(float)   # (d,)

        abs_err.append(np.linalg.norm(t - e, ord=1))
        ref_mag.append(np.linalg.norm(t,     ord=1))

        nt, ne = np.linalg.norm(t), np.linalg.norm(e)
        dot    = float(np.dot(t, e))
        cos_vals.append(dot / (nt * ne) if nt > 1e-12 and ne > 1e-12 else 1.0)

    mae = float(np.mean(abs_err))
    if normalize:
        denom = float(np.mean(ref_mag))
        mae = mae / denom if denom > 1e-12 else mae

    return mae, float(np.mean(cos_vals)), None, None



# ---------------------------------------------------------------------
# 2) Uniform-grid, *additive rotation term already baked into drifts*
# ---------------------------------------------------------------------
def compute_drift_metrics_with_rotation(true_drift, est_drift, d,
                                        grid_size=5.0, resolution=50,
                                        normalize=True):
    if d == 1:
        xs = np.linspace(-grid_size, grid_size, resolution)
        points = [np.array([xs_i]) for xs_i in xs]
    elif d == 2:
        xs = np.linspace(-grid_size, grid_size, resolution)
        ys = np.linspace(-grid_size, grid_size, resolution)
        X, Y = np.meshgrid(xs, ys)
        points = [np.array([x, y]) for x, y in zip(X.ravel(), Y.ravel())]
    else:
        points = np.random.uniform(-grid_size, grid_size, size=(1000, d))

    abs_err, ref_mag, cos_vals = [], [], []
    for x in points:
        t = np.ravel(true_drift(x)).astype(float)   # always (d,)
        e = np.ravel(est_drift(x)).astype(float)

        abs_err.append(np.linalg.norm(t - e, ord=1))
        ref_mag.append(np.linalg.norm(t,     ord=1))

        nt, ne = np.linalg.norm(t), np.linalg.norm(e)
        dot = (t * e).sum()                         # element-wise, no @
        cos_vals.append(dot / (nt * ne) if nt > 1e-12 and ne > 1e-12 else 1.0)

    mae = float(np.mean(abs_err))
    if normalize:
        denom = float(np.mean(ref_mag))
        mae = mae / denom if denom > 1e-12 else mae

    return mae, float(np.mean(cos_vals)), None, None




def main(args: argparse.Namespace) -> None:
    """
    Train a model, (optionally) run SB refinement, then evaluate / log.
    """
    run_dir = os.path.join(args.root, f"p0-{args.p0}", args.dataset)
    data_dir = os.path.join(run_dir, "data")
    method_dir = os.path.join(run_dir, args.method_tag)  # per-method subfolder
    os.makedirs(method_dir, exist_ok=True)
    # ------------------------------------------------------------------
    #  Initialise
    # ------------------------------------------------------------------
    key = jax.random.PRNGKey(args.seed)
    np.random.seed(args.seed)

    config          = yaml.safe_load(open('config.yaml'))
    jkonet_config   = yaml.safe_load(open('config-jkonet-extra.yaml'))
    config.update(jkonet_config)

    batch_size  = config['train']['batch_size']
    if args.epochs is None:  # user gave no flag → fall back
        epochs = config['train']['epochs']
    else:  # user specified something
        epochs = args.epochs  # could be 0
    eval_freq   = config['train']['eval_freq']
    save_locally = config['train']['save_locally']

    #  wandb ----------------------------------------------------------------
    if args.wandb:
        wandb.init(project=wandb_config['project'], config=config)
        tag = getattr(args, 'method_tag', 'vanilla')
        wandb.run.name = f"{args.solver}.{args.dataset}.{tag}.seed{args.seed}"

    #  Data & model ---------------------------------------------------------
    dataset_eval = PopulationEvalDataset(
        key, args.dataset, str(args.solver),
        config['metrics']['wasserstein_error'],
        args.eval, dt=args.dt,
        root=args.root, p0=args.p0
    )

    model = get_model(args.solver, config,
                      dataset_eval.data_dim, dataset_eval.dt)
    state = model.create_state(key)

    # dataset_train = model.load_dataset(args.dataset, root=args.root, p0=args.p0)

    os.environ['EXP_DATA_ROOT'] = args.root
    os.environ['EXP_P0'] = args.p0

    dataset_train = model.load_dataset(args.dataset)

    torch.manual_seed(args.seed)
    loader_train = DataLoader(
        dataset_train,
        batch_size=batch_size if batch_size > 0 else len(dataset_train),
        shuffle=True,
        collate_fn=numpy_collate)
    loader_val = DataLoader(
        dataset_eval,
        batch_size=len(dataset_eval),
        shuffle=False,
        collate_fn=numpy_collate)

    print(f"Training {args.solver} on {args.dataset} (seed {args.seed}) "
          f"for {epochs} epochs.")

    # ----------------------------------------------------------------------
    #  TRAIN JKONET*
    # ----------------------------------------------------------------------
    progress_bar = tqdm(range(1, epochs + 1))
    train_step = jax.jit(model.train_step) if epochs > 1 else model.train_step

    for epoch in progress_bar:
        if epochs == 0:
            print(f"[train] epochs=0  →  Skipping JKONet* training; "
                  f"proceeding directly to SB refinement.")
        else:
            loss = 0.0
            t0 = time()
            for batch in loader_train:
                l, state = train_step(state, batch)
                loss += l
            t1 = time()

            progress_bar.desc = f"Epoch {epoch} | Loss={loss/len(loader_train):.4g}"
            if args.wandb:
                wandb.log({'epoch': epoch,
                           'time': t1 - t0,
                           'loss': float(loss) / len(loader_train)})

    # ======================================================================
    #  (2 b)  OPTIONAL SCHRÖDINGER-BRIDGE REFINEMENT
    # ======================================================================
    use_refined = False           # will flip to True if SB is run
    refined_drift_fn = None       # placeholder

    if epochs == 0:
        rng = np.random.default_rng(args.seed)
        init_sigma2 = args.diffusivity * 10 ** (rng.uniform(-1,1))
        # if rng.random() < 0.5:
        #     # sample below
        #     init_sigma2 = rng.uniform(0.1 * args.diffusivity, args.diffusivity)
        # else:
        #     # sample above
        #     init_sigma2 = rng.uniform(args.diffusivity, 10.0 * args.diffusivity)
        # init_sigma2 = args.init_sigma2
    else:
        init_sigma2 = 2 * model.get_beta(state)
    # --- where to save iter metrics/plots ---
    inference_dir = os.path.join('out', 'plots', args.dataset, args.method_tag)
    os.makedirs(inference_dir, exist_ok=True)

    # --- build true drift and true sigma^2 for diagnostics ---
    true_sigma2 = float(args.diffusivity)  # you can override via flag if desired
    true_drift_fn = _make_true_drift_fn(args.potential, args.rotation)
    if args.sb_iters > 0:
        print(f"\n=== SB refinement: {args.sb_iters} outer iterations ===")

        out = sb_refine(
            model=model,
            state=state,
            eval_dataset=dataset_eval,
            init_sigma2=init_sigma2,
            n_outer=args.sb_iters,
            fix_diffusion=args.fix_diffusion,
            nn_width=args.sb_nn_width,
            nn_depth=args.sb_nn_depth,
            nn_lr=args.sb_nn_lr,
            nn_epochs=args.sb_nn_epochs,
            nn_conservative=True,
            nn_activation=args.activation,
            save_dir=method_dir,
        )
        refined_drift_fn, _, sigma2_refined = out
        use_refined = True

        # ----------------------------------------------------------------------
    #  EVALUATION  (single pass on final model / refined drift)
    # ----------------------------------------------------------------------
    key, key_eval = jax.random.split(key)
    init_pp = next(iter(loader_val))

    potential = model.get_potential(state)
    interaction = model.get_interaction(state)

    if use_refined and sigma2_refined is not None:
        beta_eval = 0.5 * float(np.clip(sigma2_refined, 1e-8, 1000000.0))
        print(f"estimated diffusivity (σ², refined): {2 * beta_eval:.4g}")
    else:
        # beta_raw = model.get_beta(state)
        beta_eval =  (model.get_beta(state))
        print(f"estimated diffusivity (σ², JKONet*): {2 * beta_eval:.4g}")
            # (
            #
            # _safe_scalar(beta_raw, 0.1, 1e-6, 1000000.0)))


    print('estimated diffusivity (σ²):', 2 * beta_eval)


    try:
        predictions = get_SDE_predictions(
            str(args.solver),
            dataset_eval.dt, dataset_eval.T, 1,
            potential, beta_eval, interaction,
            key_eval, init_pp)
    except Exception as e:
        print(f"[guard] get_SDE_predictions failed ({e}); aborting eval.")
        return

    # ----------------------------------------------------------------------
    # Build a single NumPy-friendly drift function for downstream use
    # ----------------------------------------------------------------------
    est_drift_fn = _get_estimated_drift_fn(
        model=model,
        state=state,
        use_refined=use_refined,
        refined_drift_fn=refined_drift_fn
    )

    # Save inference bundle alongside experiment results so we can compute
    # any metric later without retraining.
    inference_dir = os.path.join('out', 'plots', args.dataset, args.method_tag)
    _save_inference_bundle(
        method_dir,
        dim=dataset_eval.data_dim,
        grid_size=args.grid_size,
        resolution=51,
        est_sigma2=2 * beta_eval,   # store σ²
        est_drift_fn=est_drift_fn,
        seed=args.seed,
        n_random=100,
        init_sigma2=init_sigma2
    )
    print(f"[inference] saved bundle to {method_dir}/inference_bundle.npz")



if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--solver', '-s',
        type=EnumMethod,
        choices=list(EnumMethod),
        default=EnumMethod.JKO_NET_STAR_POTENTIAL,
        help=f"""Name of the solver to use.""",
    )

    parser.add_argument(
        '--root', type=str, default='main_experiments',
        help="Root directory where runs are stored (default: main_experiments)."
    )
    parser.add_argument(
        '--p0', type=str, default='gmm', choices=['gmm', 'gibbs', 'unif'],
        help="Initial distribution tag; results under p0-<p0>/ (default: gmm)."
    )

    parser.add_argument(
        '--dataset', '-d',
        type=str,
        help=f"""Name of the dataset to train the model on. The name of the dataset should match the name of the directory generated by the `data_generator.py` script.""",
    )

    parser.add_argument(
        '--eval',
        type=str,
        default='train_data',
        choices=['train_data', 'test_data'],
        help=f"""Option to test fit on test data or train data (e.g., for debugging purposes).""",
    )

    parser.add_argument('--wandb', action='store_true',
                        help='Option to run with activated wandb.')

    parser.add_argument('--debug', action='store_true',
                        help='Option to run in debug mode.')
    parser.add_argument('--fix_diffusion', action='store_true',
                        help='Set True to fix diffusion across all iterations of SB refinement.')

    parser.add_argument('--epochs', type=int, help='Number of epochs to train the model.')
    parser.add_argument('--potential', type=str, default='none',
                        choices=list(potentials_all.keys()) + ['none'],
                        help="Name of the potential energy to use.")
    parser.add_argument('--rotation', type=str, default='none',
                        choices = list(rotations_all.keys()) + ['none'],
                        help = "Name of the divergence-free drift (if any).")
    parser.add_argument("--grid_size", type=float, default=5.0, help="grid_size for L1 drift error computation")
    parser.add_argument("--diffusivity", type=float, default=0.2, help="true diffusivity")
    parser.add_argument("--init_sigma2", type=float, default=1.0, help="diffusivity prior")
    parser.add_argument('--sb-iters', type=int, default=30,
                        help="Number of Schrödinger-bridge outer iterations (0 → skip).")
    parser.add_argument(
        "--activation", type=str, default="silu",
        help="Activation function for NN drift (softplus, silu, relu, tanh, gelu)"
    )
    parser.add_argument('--sb-nn-width', type=int, default=128)
    parser.add_argument('--sb-nn-depth', type=int, default=2)
    parser.add_argument('--sb-nn-lr', type=float, default=3e-3)
    parser.add_argument('--sb-nn-epochs', type=int, default=500)
    parser.add_argument('--sb-nn-conservative', action='store_true',
                        help='Use scalar potential φ and drift = -grad(φ).')
    # in __main__ argparse section
    parser.add_argument('--method-tag', type=str, default='vanilla',
                        help='Subfolder tag to store results under out/plots/<dataset>/<method-tag>')
    # reproducibility
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Set seed for the run'
    )
    parser.add_argument('--dt', type=float, default=0.01,
                        help="Euler–Maruyama step used for evaluation/prediction.")

    args = parser.parse_args()

    # set debug mode
    if args.debug:
        print('Running in DEBUG mode.')
        jax.config.update('jax_disable_jit', True)

    main(args)
