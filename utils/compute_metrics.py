import json
import numpy as np
from pathlib import Path
import re


import jax
import jax.numpy as jnp
from utils.functions import potentials_all as _POT_ALL


# ------------------------- config/mappings -------------------------
FOLDER2POT = {
    "quadratic": "poly",
    "poly": "poly",
    "bohachevsky": "bohachevsky",
    "oakley_ohagan": "oakley_ohagan",
    "styblinski_tang": "styblinski_tang",
    "wavy_plateau": "wavy_plateau",
}


def parse_run_folder(name: str, prefix: str | set[str] | None = None):
    """
    Accept BOTH old and new run-folder names.

    Old (back-compat):
        3_margs_langevin_(<prefix>_)?<pot>_steps-<k>_diff-<sig>_seed-<seed>

    New (current layout):
        3_margs_langevin_<pot>_diff-<sig>_seed-<seed>

    Returns (pot, step, sigma2, seed) or None.
    """
    pat_old = (
        r"^3_margs_langevin_(?:(?P<prefix>[^_]+)_)?"
        r"(?P<pot>.+?)_steps-(?P<steps>\d+)_diff-(?P<sig>[\d.]+)_seed-(?P<seed>\d+)$"
    )
    m = re.match(pat_old, name)
    if m:
        if prefix is not None:
            allowed = {prefix} if isinstance(prefix, str) else set(prefix)
            pref = m.group("prefix")
            if pref is None or pref not in allowed:
                return None
        pot_raw = m.group("pot")
        pot = FOLDER2POT.get(pot_raw, pot_raw)
        return pot, int(m.group("steps")), float(m.group("sig")), int(m.group("seed"))

    pat_new = (
        r"^3_margs_langevin_"
        r"(?P<pot>.+?)_diff-(?P<sig>[\d.]+)_seed-(?P<seed>\d+)$"
    )
    m = re.match(pat_new, name)
    if m:
        pot_raw = m.group("pot")
        pot = FOLDER2POT.get(pot_raw, pot_raw)
        # interpret missing steps as 0 for compatibility
        return pot, 0, float(m.group("sig")), int(m.group("seed"))

    return None

def load_bundle(folder: Path):
    """Load inferred grid & drift from inference_bundle.npz and reshape to mesh."""
    npz_path = folder / "inference_bundle.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing file: {npz_path}")
    data = np.load(npz_path)
    gp, gd = data["grid_points"], data["grid_drift"]

    # Optional meta (unused except for debugging)
    meta_path = folder / "inference_meta.json"
    if meta_path.exists():
        try:
            _ = json.loads(meta_path.read_text())
        except Exception:
            pass

    x_unique, y_unique = np.unique(gp[:, 0]), np.unique(gp[:, 1])
    X, Y = np.meshgrid(np.sort(x_unique), np.sort(y_unique))
    U, V = np.zeros_like(X), np.zeros_like(Y)

    # Map (x,y) -> (u,v), robust to tiny float jitter via rounding.
    def key(a, b): return (round(float(a), 8), round(float(b), 8))
    drift_map = {key(x, y): (u, v) for (x, y), (u, v) in zip(gp, gd)}
    for i in range(Y.shape[0]):
        for j in range(X.shape[1]):
            uv = drift_map.get(key(X[i, j], Y[i, j]))
            if uv is None:
                idx = np.argmin((gp[:, 0] - X[i, j]) ** 2 + (gp[:, 1] - Y[i, j]) ** 2)
                uv = gd[idx]
            U[i, j], V[i, j] = uv
    return dict(X=X, Y=Y, U=U, V=V, speed=np.hypot(U, V))

def load_estimated_sigma2(bundle_dir: Path) -> float | None:
    meta = bundle_dir / "inference_meta.json"
    if not meta.exists():
        return None
    try:
        with open(meta, "r") as f:
            j = json.load(f)
        for k in ["sigma2", "estimated_sigma2", "est_sigma2", "sigma_sq"]:
            if k in j:
                return float(j[k])
    except Exception:
        pass
    return None

def get_gt_callable(potential: str):
    """Return (V, drift) where drift(v) = -∇V(v) from utils.functions."""
    if potential == "quadratic":
        poly = _POT_ALL["poly"]
        def V(v): return poly(v, {2: 5.0})
    else:
        if potential not in _POT_ALL:
            return None, None
        V = _POT_ALL[potential]
    dV = jax.grad(V)
    def drift(v): return -dV(v)
    return V, drift

def make_gt_bundle(base_dir: Path, potential: str, borrow_from):
    """
    Build a 'bundle' (X,Y,U,V,speed,_gtZ) from -∇Ψ on the same grid as one of
    the existing method subfolders under base_dir.
    """
    grid = None
    for m in borrow_from:
        try:
            b = load_bundle(base_dir / m)
            grid = (b["X"], b["Y"])
            break
        except Exception:
            pass
    if grid is None:
        raise RuntimeError(
            f"No inferred bundle in {base_dir}/({ '|'.join(borrow_from) }) "
            "to borrow a grid for panel 'gt'."
        )
    X, Y = grid
    Z, Ugt, Vgt = eval_gt_on_grid(potential, X, Y)
    if Z is None:
        msg = "Ground-truth requires JAX + utils.functions."
    return dict(X=X, Y=Y, U=Ugt, V=Vgt, speed=np.hypot(Ugt, Vgt), _gtZ=Z)

def eval_gt_on_grid(potential: str, X: np.ndarray, Y: np.ndarray):
    """Evaluate Ψ and -∇Ψ on a given grid (requires JAX + utils.functions)."""
    Vfun, driftfun = get_gt_callable(potential)
    if Vfun is None:
        return None, None, None
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    Vv = jax.vmap(lambda xy: Vfun(jnp.array(xy)))
    Dv = jax.vmap(lambda xy: driftfun(jnp.array(xy)))
    Z = np.array(Vv(pts)).reshape(X.shape)
    Z = Z - Z.min()  # set min Ψ = 0 for readability
    UV = np.array(Dv(pts)).reshape(X.shape + (2,))
    Ugt, Vgt = UV[..., 0], UV[..., 1]
    return Z, Ugt, Vgt

def compute_metrics(Ue, Ve, Ugt, Vgt, eps=1e-12):
    """
    Normalized drift MAE (L1) and cosine similarity (averaged over grid).
    - MAE_L1 = mean(|ΔU| + |ΔV|)
    - denom  = mean(|Ugt| + |Vgt|) + eps
    """
    dU, dV = Ue - Ugt, Ve - Vgt
    mae_l1 = np.mean(np.abs(dU) + np.abs(dV))
    gt_l1  = np.mean(np.abs(Ugt) + np.abs(Vgt)) + eps
    nmae   = mae_l1 / gt_l1

    num = Ue * Ugt + Ve * Vgt
    den = (np.sqrt(Ue**2 + Ve**2) * np.sqrt(Ugt**2 + Vgt**2) + eps)
    cos = np.mean(num / den)
    return nmae, cos

def compute_diff_mae_for_run(run_dir: Path, method_subdir: str, parsed_info=None) -> dict | None:
    """
    Diffusivity MAE for one run/method:
      value = | true_sigma^2 (from folder name) - est_sigma^2 (from inference_meta.json) |
    Returns dict with potential, step, seed, value — or None if unavailable.
    """
    parsed = parsed_info if parsed_info is not None else parse_run_folder(run_dir.name)
    if not parsed:
        return None
    pot, step, true_sigma2, seed = parsed

    bundle_dir = run_dir / method_subdir
    # try to read the estimated sigma^2 from the method's meta
    est_sigma2 = load_estimated_sigma2(bundle_dir)
    if est_sigma2 is None:
        return None

    return {
        "potential": pot,
        "step": step,
        "seed": seed,
        "value": float(abs(float(true_sigma2) - float(est_sigma2))),
    }

def sem(x):
    x = np.asarray(x, float)
    return float(np.std(x, ddof=1) / np.sqrt(len(x))) if x.size > 1 else 0.0