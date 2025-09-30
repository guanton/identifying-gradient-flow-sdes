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
This module provides a set of plotting utilities for visualizing data and model predictions using `matplotlib`. 
The primary functionalities include plotting couplings between particles, generating level curves, visualizing model 
predictions, creating heatmaps, comparing execution times of models, and plotting loss comparisons across different 
model parameters.

Functions
-------------
- ``plot_couplings``
    Visualizes connections between two sets of points (circles and crosses) with lines whose widths are proportional to the weights.

- ``domain_from_data``
    Computes the domain boundaries for plotting based on the data.

- ``grid_from_domain``
    Generates a grid of points within a specified domain for visualization purposes.

- ``plot_level_curves``
    Plots the level curves of a given function over a specified domain.

- ``plot_predictions``
    Visualizes the predicted and ground truth particle positions for different timesteps.

- ``colormap_from_config``
    Creates a custom colormap from a configuration dictionary.

- ``plot_heatmap``
    Plots a heatmap of values over a 2D grid.

- ``plot_boxplot_comparison_models``
    Creates a boxplot to compare execution times of different models.

- ``plot_comparison_models``
    Compares two sets of model errors, with optional insets for detailed views.

- ``plot_loss``
    Plots the loss values for different models over varying parameter values.


Usage Example
-------------
To plot couplings between two sets of points:
    
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> data = np.array([[0., 0., 0., 1., 5., 0.5], [1., 0., 2., 2., 5., 0.5]])
    >>> fig, ax = plot_couplings(data)
    >>> plt.show()
"""

import imageio
from io import BytesIO
import os
from pathlib import Path
import jax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import yaml
from typing import Tuple, Optional, Callable, Dict, List, Literal, Union


def create_marginal_gif(data, measurement_times, V, sigma, gif_filename='marginals.gif',
                        bins=30, sde_title=None, show_gibbs=False, duration=0.5, output_dir = ''):
    """
    Create a GIF of the marginal distributions over all measurement times.

    Parameters:
        data (np.ndarray): Array of shape (num_trajectories, n_measurements, d).
        measurement_times (array-like): List/array of measurement times.
        V (function): Potential function (wrapped) for computing the Gibbs density.
        sigma (float): Diffusivity constant.
        gif_filename (str): Output filename for the GIF.
        bins (int): Number of bins for 1D histograms.
        sde_title (str, optional): SDE title to display on each frame.
        show_gibbs (bool): If True, overlay the Gibbs density.
        duration (float): Duration (in seconds) per frame in the GIF.
    """
    images = []
    measurement_times = np.array(measurement_times)
    # Compute global axis limits.
    d = data.shape[2]
    if d == 1:
        x_min, x_max = compute_axes_limits(data, dim=1)
        y_min = y_max = None
    elif d >= 2:
        x_min, x_max, y_min, y_max = compute_axes_limits(data, dim=2)

    for t in measurement_times:
        plt.figure()
        plot_time_marginal(data, t, measurement_times, V, sigma, bins=bins,
                           x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                           sde_title=sde_title, show_gibbs=show_gibbs, display=False)
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        image = imageio.imread(buf)
        images.append(image)

    imageio.mimsave(os.path.join(output_dir, gif_filename), images, duration=duration, loop=0)
    print(f"GIF saved to {gif_filename}")


def plot_time_marginal(data, measurement_time, measurement_times, V, sigma, bins=30,
                       x_min=None, x_max=None, y_min=None, y_max=None,
                       sde_title=None, show_gibbs=True, display=True):
    """
    Plot the marginal distribution at a given measurement time with an overlay of the Gibbs distribution.

    Parameters:
        data (np.ndarray): Array of shape (num_trajectories, n_measurements, d) containing trajectory data.
        measurement_time (float): The measurement time to plot.
        measurement_times (array-like): Array of measurement times corresponding to the data.
        V (function): Potential function (wrapped to accept numpy arrays) used to compute the Gibbs density.
        sigma (float): Diffusivity constant.
        bins (int): Number of bins for histogram (for 1D data).
        x_min, x_max (float, optional): Limits for the x-axis.
        y_min, y_max (float, optional): Limits for the y-axis (for d>=2).
        sde_title (str, optional): A title string for the SDE to display in the plot.
        show_gibbs (bool): If True, overlay the computed Gibbs density.

    Notes:
        - For 1D data, the function plots a histogram (with consistent x-axis) and overlays a line for the Gibbs density.
        - For 2D data, it creates a scatter plot (first two dimensions) and overlays contour lines of the Gibbs density.
    """
    measurement_times = np.array(measurement_times)
    time_index = np.argmin(np.abs(measurement_times - measurement_time))
    marginals = data[:, time_index, :]
    d = marginals.shape[1]

    # Compute global axis limits if not provided.
    if d == 1:
        if x_min is None or x_max is None:
            x_min, x_max = compute_axes_limits(data, dim=1)
    elif d >= 2:
        if x_min is None or x_max is None or y_min is None or y_max is None:
            x_min, x_max, y_min, y_max = compute_axes_limits(data, dim=2)

    title_str = sde_title if sde_title is not None else f"Overdamped Langevin: dXₜ = -∇V(Xₜ) dt + {sigma} dWₜ"

    if d == 1:
        plt.figure(figsize=(6, 5))
        # Plot the histogram.
        plt.hist(marginals[:, 0], bins=bins, density=True, alpha=0.7, color='steelblue', range=(x_min, x_max))
        # Overlay the Gibbs density.
        if show_gibbs:
            x_grid = np.linspace(x_min, x_max, 200)
            # Compute the unnormalized Gibbs density: exp(-2V(x)/sigma²)
            density = np.array([np.exp(-2 * V(np.array([x])) / sigma ** 2) for x in x_grid])
            # Normalize the density.
            Z = np.trapz(density, x_grid)
            density = density / Z
            plt.plot(x_grid, density, 'r-', lw=2, label='Gibbs Density')
            plt.legend()
        plt.xlabel("x")
        plt.ylabel("Density")
        plt.title(f"1D Marginal at t = {measurement_times[time_index]:.3f}\n{sde_title}")
    elif d == 2:
        plt.figure(figsize=(6, 5))
        plt.scatter(marginals[:, 0], marginals[:, 1], alpha=0.7, color='darkorange')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Marginal at t = {measurement_times[time_index]:.3f}\n{sde_title}")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        # Overlay Gibbs density contours.
        if show_gibbs:
            xx = np.linspace(x_min, x_max, 100)
            yy = np.linspace(y_min, y_max, 100)
            X, Y = np.meshgrid(xx, yy)

            # Compute log Gibbs density on grid
            def gibbs_log_density(x, y):
                return -2 * V(np.array([x, y])) / sigma ** 2

            vec_gibbs_log = np.vectorize(gibbs_log_density)
            logZ = vec_gibbs_log(X, Y)
            max_log = np.max(logZ)
            Z = np.exp(logZ - max_log)
            # Normalize by double integration.
            Z_norm = np.trapz(np.trapz(Z, xx, axis=1), yy)
            Z = Z / Z_norm
            levels = np.linspace(np.nanmin(Z), np.nanmax(Z), 6)
            cs = plt.contour(X, Y, Z, levels=levels, colors='k', linestyles='--')
            plt.clabel(cs, inline=True, fontsize=8)
        plt.grid(True)
    else:
        print('Not implemented')
    plt.tight_layout()
    if display:
        plt.show()

def compute_axes_limits(data, dim=1):
    """
    Compute axis limits automatically from the entire data array.

    Parameters:
        data (np.ndarray): Array of shape (num_trajectories, n_measurements, d).
        dim (int): 1 for 1D data, 2 for 2D data.

    Returns:
        For 1D: (x_min, x_max).
        For 2D: (x_min, x_max, y_min, y_max).
    """
    if dim == 1:
        x_vals = data[:, :, 0]
        x_min = np.min(x_vals)
        x_max = np.max(x_vals)
        margin = 0.1 * (x_max - x_min)
        return x_min - margin, x_max + margin
    elif dim >= 2:
        x_vals = data[:, :, 0]
        y_vals = data[:, :, 1]
        x_min = np.min(x_vals)
        x_max = np.max(x_vals)
        y_min = np.min(y_vals)
        y_max = np.max(y_vals)
        x_margin = 0.1 * (x_max - x_min)
        y_margin = 0.1 * (y_max - y_min)
        return x_min - x_margin, x_max + x_margin, y_min - y_margin, y_max + y_margin

def plot_couplings(data: np.ndarray) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots circles at x coordinates, crosses at y coordinates,
    and connects them with lines whose widths are proportional to weights.

    Parameters
    ----------
    data : np.ndarray
        An array of shape (n, 6) where each row contains:

        - x0, x1 (coordinates of the circle),
        - y0, y1 (coordinates of the cross),
        - time label
        - w (weight for line width).

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        The matplotlib figure and axis objects of the plot.

    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> data = np.array([[0. , 0. , 0. , 1. , 5. , 0.5],
    ...             [1. , 0. , 2. , 2. , 5. , 0.5]])
    >>> fig, ax = plot_couplings(data)
    >>> plt.show()  # This will display the plot

    .. toggle:: Click to toggle plot

        .. image:: ../_static/plotting_documentation/plot_couplings.png
           :align: center
           :alt: Example plot showing circles connected to crosses with weighted lines.
    """
    # Extract coordinates and weights
    weights = data[:, -1]
    x_coords = data[:, :(data.shape[1] - 1) // 2]
    y_coords = data[:, (data.shape[1] - 1) // 2:-2]

    # Normalize weights for line width between a minimum and a maximum
    line_widths = 2 * (weights / weights.max())  # Normalize and scale line width by weight

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot circles at x positions
    ax.scatter(x_coords[:, 0], x_coords[:, 1], c='blue', s=100, edgecolors='black', label='Circles', marker='o')

    # Plot crosses at y positions
    ax.scatter(y_coords[:, 0], y_coords[:, 1], c='red', s=100, label='Crosses', marker='x')

    # Draw lines connecting circles and crosses
    for x, y, lw in zip(x_coords, y_coords, line_widths):
        ax.plot([x[0], y[0]], [x[1], y[1]], 'gray', linewidth=lw)

    # Adding labels and title for clarity
    # ax.set_xlabel('X coordinate')
    # ax.set_ylabel('Y coordinate')
    # ax.set_title('Connections Between Points')
    # ax.legend()

    return fig, ax


def domain_from_data(data: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Calculate the domain boundaries from the data for plotting purposes.

    Parameters
    ----------
    data : np.ndarray
        An array where each row contains at least two coordinates (x, y).

    Returns
    -------
    Tuple[Tuple[float, float], Tuple[float, float]]
        A tuple containing two tuples:

        - The minimum (x_min, y_min) and
        - The maximum (x_max, y_max) coordinates, with additional padding.

    Example
    -------
    >>> import numpy as np
    >>> data = np.array([[0., 0.], [2., 2.]])
    >>> domain_from_data(data)
    ((-2.0, -2.0), (4.0, 4.0))

    """
    # set max and min values
    x_min = np.amin(data, axis=0)[:, 0].min() - 2.0
    x_max = np.amax(data, axis=0)[:, 0].max() + 2.0

    y_min = np.amin(data, axis=0)[:, 1].min() - 2.0
    y_max = np.amax(data, axis=0)[:, 1].max() + 2.0

    return ((x_min, y_min), (x_max, y_max))


def grid_from_domain(
    domain: Tuple[Tuple[float, float], Tuple[float, float]],
    n_samples: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a grid of points within a specified domain.

    Parameters
    ----------
    domain : Tuple[Tuple[float, float], Tuple[float, float]]
        The domain within which to create the grid. It is a tuple containing two tuples:

        - The lower bounds (x_min, y_min) of the domain.
        - The upper bounds (x_max, y_max) of the domain.
    n_samples : int, optional
        The number of samples (grid points) along each axis. Default is 100.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - x : np.ndarray
            The x-coordinates of the grid points.
        - y : np.ndarray
            The y-coordinates of the grid points.
        - grid : np.ndarray
            The grid of points in (x, y) space. If the domain has more than 2 dimensions, extra dimensions are filled
            with zeros as to project into the other dimensions.

    Example
    -------
    >>> import numpy as np
    >>> domain = ((-2.0, -2.0), (4.0, 4.0))
    >>> x, y, grid = grid_from_domain(domain)
    >>> x.shape, y.shape, grid.shape
    ((100, 100), (100, 100), (10000, 2))

    """
    # create grid
    x, y = np.meshgrid(np.linspace(domain[0][0], domain[1][0], n_samples),
                       np.linspace(domain[0][1], domain[1][1], n_samples))
    grid = np.vstack((x.ravel(), y.ravel())).T

    if len(domain[0]) > 2:
        grid = np.concatenate((grid, np.zeros((grid.shape[0],
                                               len(domain[0]) - 2))), axis=1)

    return x, y, grid


def plot_level_curves(
    function: Callable[[np.ndarray], np.ndarray],
    domain: Tuple[Tuple[float, float], Tuple[float, float]],
    n_samples: int = 100,
    dimensions: int = 2,
    save_to: Optional[str] = None
)-> plt.Figure:
    """
    Plot level curves of a function over a specified domain.

    Parameters
    ----------
    function : Callable[[np.ndarray], np.ndarray]
        A function that takes a numpy array of input values and returns a scalar value.
        The function is expected to be vectorized over the input.
    domain : Tuple[Tuple[float, float], Tuple[float, float]]
        The domain over which to plot the function. It is a tuple containing:
        - The lower bounds (x_min, y_min) of the domain.
        - The upper bounds (x_max, y_max) of the domain.
    n_samples : int, optional
        The number of samples (grid points) along each axis. Default is 100.
    dimensions : int, optional
        The number of dimensions of the function output. Default is 2.
    save_to : Optional[str], default=None
        Directory path where plots should be saved. If None, no plots will be saved.

    Returns
    -------
    plt.Figure
        The matplotlib figure object containing the plot.

    Example
    -------
    Here is an example of how to use `plot_level_curves` to visualize the Styblinski-Tang function:

    .. code-block:: python

        import jax.numpy as jnp

        # Define the Styblinski-Tang function
        def styblinski_tang(v: jnp.ndarray) -> jnp.ndarray:
            u = jnp.square(v)
            return 0.5 * jnp.sum(jnp.square(u) - 16 * u + 5 * v)

        # Define the domain for plotting
        domain = ((-4.0, -4.0), (4.0, 4.0))

        # Plot and save the level curves
        fig = plot_level_curves(
            function=styblinski_tang,
            domain=domain,
            n_samples=200
        )

        # Display the plot
        plt.show()


    .. toggle:: Click to toggle plot

        .. image:: ../_static/plotting_documentation/plot_level_curves.png
           :align: center
           :alt: Example plot showing level curves of the Styblinski-Tang function.

    """
    f = jax.vmap(function)
    x, y, grid = grid_from_domain(domain, n_samples)

    # get values
    if grid.shape[1] < dimensions:
        v = np.concatenate((grid, np.zeros((grid.shape[0],
                                            dimensions - grid.shape[1]))), axis=1)
        pred = f(v)
    else:
        pred = f(grid)
    z = pred.reshape(x.shape)

    # plot energy predictions
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set(xlim=(domain[0][0], domain[1][0]), ylim=(domain[0][1], domain[1][1]))
    ax.grid(False)

    ax.contour(x, y, z, levels=15, linewidths=.5, linestyles='dotted',
               colors='k')
    ctr = ax.contourf(x, y, z, levels=15, cmap='Blues')

    if save_to is not None:
        # Save the data to a text file
        path = Path(save_to)
        os.makedirs(path.parent.absolute(), exist_ok=True)
        file = open(save_to, 'w')
        file.close()
        np.savetxt(save_to, np.column_stack(
            (x.flatten(), y.flatten(), z.flatten())), fmt='%-7.2f')


    fig.colorbar(ctr, ax=ax)

    fig.tight_layout()
    if save_to is not None:
        fig.savefig(save_to + '.png')
    return fig

import numpy as np

def _finite_bounds(arrs, pad_frac=0.05, default=(-1.0, 1.0)):
    """
    Compute finite min/max over a list of 1D arrays (or raveled ND arrays).
    Adds symmetric padding. If nothing finite, return 'default'.
    """
    if not isinstance(arrs, (list, tuple)):
        arrs = [arrs]
    vals = []
    for a in arrs:
        if a is None:
            continue
        a = np.asarray(a).ravel()
        if a.size == 0:
            continue
        good = np.isfinite(a)
        if np.any(good):
            vals.append(a[good])
    if not vals:
        return default
    v = np.concatenate(vals)
    lo, hi = float(v.min()), float(v.max())
    if not np.isfinite(lo) or not np.isfinite(hi):
        return default
    if lo == hi:  # degenerate span
        width = max(1.0, abs(lo)) * pad_frac
        return (lo - width, hi + width)
    pad = pad_frac * (hi - lo)
    return (lo - pad, hi + pad)

def _report_nonfinite(name, a):
    a = np.asarray(a)
    n = a.size
    if n == 0:
        print(f"[plot_predictions] {name}: EMPTY")
        return
    n_nan = np.isnan(a).sum()
    n_inf = np.isinf(a).sum()
    if n_nan or n_inf:
        print(f"[plot_predictions] {name}: size={n}, NaN={n_nan}, Inf={n_inf}")
def plot_predictions(predicted: np.ndarray,
    data_dict: Dict[int, np.ndarray],
    interval: Optional[Tuple[int, int]],
    model: str,
    save_to: Optional[str] = None,
    n_particles: int = 200
) -> plt.Figure:
    """
    Plot predictions and ground truth data for each timestep (2D).
    Uses _finite_bounds/_report_nonfinite to avoid NaN/Inf axis limits.
    """

    # ---- choose time window and synchronize with 'predicted' ----
    # ---- choose time window and synchronize with 'predicted' ----
    if interval is None:
        # default end = min( last predicted frame, last data frame )
        last_data_t = max(data_dict.keys()) if len(data_dict) else 0
        start, end = 0, min(predicted.shape[0] - 1, last_data_t)
    else:
        start, end = int(interval[0]), int(interval[1])

    T_pred = int(predicted.shape[0])
    if T_pred <= 0:
        raise ValueError("predicted has zero timesteps.")

    # clip both ends into [0, T_pred-1]
    start = int(np.clip(start, 0, T_pred - 1))
    end = int(np.clip(end, 0, T_pred - 1))

    # if empty after clipping, collapse to the nearest valid single frame
    if end < start:
        # if the request was entirely to the right, use the last frame; if to the left, the first.
        # after np.clip above, both cases reduce to: choose the closer of start/end
        start = end = min(max(start, 0), T_pred - 1)

    timesteps = list(range(start, end + 1))

    # slice predicted to the requested time window
    predicted_win = predicted[start:end + 1]

    # how many particles can we show
    # (respect both predicted and any available data_dict[t])
    max_particles_data = min(
        (data_dict[t].shape[0] for t in timesteps if t in data_dict),
        default=n_particles
    )
    min_particles = min(n_particles, predicted_win.shape[1], max_particles_data)

    # build dense ground‑truth tensor (zeros for missing t, if any)
    D = predicted.shape[2]  # num_dimensions; we assume D >= 2
    data = np.zeros((len(timesteps), min_particles, D), dtype=float)
    for i, t in enumerate(timesteps):
        if t in data_dict:
            data[i, :, :] = data_dict[t][:min_particles, :]

    # restrict to common particle count
    data          = data[:, :min_particles, :]
    predicted_win = predicted_win[:, :min_particles, :]

    # ---- report non‑finites (diagnostic only) ----
    _report_nonfinite("data(x)",         data[..., 0])
    _report_nonfinite("data(y)",         data[..., 1])
    _report_nonfinite("predicted(x)",    predicted_win[..., 0])
    _report_nonfinite("predicted(y)",    predicted_win[..., 1])

    # ---- robust axis limits (never NaN/Inf) ----
    x_min, x_max = _finite_bounds([data[..., 0], predicted_win[..., 0]], pad_frac=0.05, default=(-1.0, 1.0))
    y_min, y_max = _finite_bounds([data[..., 1], predicted_win[..., 1]], pad_frac=0.05, default=(-1.0, 1.0))

    # ---- plotting ----
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))

    colors = yaml.safe_load(open('style.yaml'))
    c_data = clr.LinearSegmentedColormap.from_list(
        'Greys', [colors['groundtruth']['light'], colors['groundtruth']['dark']],
        N=data.shape[0]
    )
    c_pred = clr.LinearSegmentedColormap.from_list(
        'Blues', [colors[model]['light'], colors[model]['dark']],
        N=predicted_win.shape[0]
    )

    # ground truth
    for i, t in enumerate(timesteps):
        x, y = data[i, :, 0], data[i, :, 1]
        ax.scatter(
            x, y,
            edgecolors=[c_data(i)], facecolor='none',
            label=f'data, t={t}', marker=colors['groundtruth']['marker']
        )
        if save_to is not None:
            np.savetxt(f"{save_to}-data-{t}.txt", np.column_stack((x, y)), fmt='%-7.2f')

    # predicted
    for i, t in enumerate(timesteps):
        x, y = predicted_win[i, :, 0], predicted_win[i, :, 1]
        ax.scatter(
            x, y,
            c=[c_pred(i)],
            label=f'predicted, t={t}', marker=colors[model]['marker']
        )
        if save_to is not None:
            np.savetxt(f"{save_to}-predicted-{t}.txt", np.column_stack((x, y)), fmt='%-7.2f')

    ax.legend(bbox_to_anchor=(0.5, 1.25), fontsize='medium',
              loc='upper center', ncol=3, columnspacing=1, frameon=False)

    fig.tight_layout()
    if save_to is not None:
        fig.savefig(f"{save_to}.png")
    return fig


def colormap_from_config(config: Dict[str, str]) -> clr.LinearSegmentedColormap:
    """
    Create a colormap from the provided configuration.

    Parameters
    ----------
    config : Dict[str, str]
        A dictionary containing 'light' and 'dark' color codes for the colormap.

    Returns
    -------
    clr.LinearSegmentedColormap
        The custom colormap created from the given colors.
    """
    light = config['light']
    dark = config['dark']
    return clr.LinearSegmentedColormap.from_list('custom', [light, dark])

def plot_heatmap(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    labels: Dict[str, str],
    title: str,
    colormap: str,
    save_to: Optional[str] = None
) -> plt.Figure:
    """
    Plot a heatmap with color mapping and save the figure and data to files if requested.

    Parameters
    ----------
    X : np.ndarray
        2D array of shape (m, n) representing the x-coordinates of the heatmap grid.
    Y : np.ndarray
        2D array of shape (m, n) representing the y-coordinates of the heatmap grid.
    Z : np.ndarray
        2D array of shape (m, n) representing the values for the heatmap.
    labels : Dict[str, str]
        Dictionary with keys 'X', 'Y', and 'Z' mapping to axis labels and colorbar label.
    title : str
        Title of the heatmap plot.
    colormap : str
        Name of the colormap to use for the heatmap.
    save_to : Optional[str], default=None
        Directory path where plots should be saved. If None, no plots will be saved.

    Returns
    -------
    plt.Figure
        The matplotlib figure object containing the plot.
    """
    fig = plt.figure()
    plt.pcolormesh(X, Y, Z, shading='auto', cmap=colormap)
    plt.colorbar(label=labels['Z'])
    plt.xlabel(labels['X'])
    plt.ylabel(labels['Y'])
    plt.xticks(X[0])
    plt.yticks(Y[:, 0])
    plt.title(title)

    if save_to is not None:
        plt.savefig(save_to + '.png')
        with open(save_to + '.csv', 'w') as file:
            file.write(f'x y z\n')
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    file.write(f'{X[i, j]:.10f} {Y[i, j]:.10f} {Z[i, j]:.10f}\n')
                file.write('\n')
    return fig

def plot_boxplot_comparison_models(
    data: List[np.ndarray],
    model_names: List[str],
    title: str,
    save_to: Optional[str] = None,
    yscale: Literal['linear', 'log'] = 'linear'
) -> plt.Figure:
    """
    Create a boxplot to compare execution times of different models.

    Parameters
    ----------
    data : List[np.ndarray]
        List of 1D arrays, each containing execution times for a model.
    model_names : List[str]
        List of names for each model, corresponding to the data list.
    title : str
        Title of the boxplot.
    save_to : Optional[str], default=None
        Directory path where plots should be saved. If None, no plots will be saved.
    yscale : Literal['linear', 'log']
        Scale type for the y-axis; 'linear' or 'log'.

    Returns
    -------
    plt.Figure
        The matplotlib figure object containing the plot.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    bp = ax.boxplot(data, patch_artist=True, meanline=True, showmeans=True)

    # Customizing the box plot colors
    style = yaml.safe_load(open('style.yaml'))
    colors = [style[model]['dark'] for model in model_names]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Adding custom labels for clarity
    ax.set_xticklabels(model_names)
    ax.set_title(title)
    ax.set_ylabel('Execution Time [s]')
    ax.set_yscale(yscale)
    plt.xticks(rotation=30, ha='right')
    fig.tight_layout()

    if save_to is not None:
        plt.savefig(save_to + '.png')
        for model in model_names:
            np.savetxt(save_to + f'-{model}.txt', data[model_names.index(model)], fmt='%.2f')

    return fig


def plot_comparison_models(
    error1: np.ndarray,
    error2: np.ndarray,
    labels: np.ndarray,
    model_names: List[str],
    title: str,
    save_to: Optional[str] = None,
    cmaps: Optional[List[str]] = None,
    insert_inset: bool = False,
    size: int = 100
) -> plt.Figure:
    """
    Plot a comparison between two sets of errors, with optional insets to highlight details.

    Parameters
    ----------
    error1 : np.ndarray
        Array of errors for the first set of predictions.
    error2 : np.ndarray
        Array of errors for the second set of predictions.
    labels : np.ndarray
        Array of labels used to group errors.
    model_names : List[str]
        List of model names for the axes labels.
    title : str
        Title of the plot.
    save_to : Optional[str], default=None
        Directory path where plots should be saved. If None, no plots will be saved.
    cmaps : Optional[List[str]]
        List of color maps to use for each label. If None, a default colormap is used.
    insert_inset : bool
        Whether to include an inset plot for detailed views.
    size : int
        Marker size for scatter points.

    Returns
    -------
    plt.Figure
        The matplotlib figure object containing the plot.
    """
    error1 = np.asarray(error1)
    error2 = np.asarray(error2)
    labels = np.asarray(labels)

    #Normalize errors
    max_error = np.nanmax(np.concatenate((error1, error2)))
    normalized_error1 = error1 / max_error
    normalized_error2 = error2 / max_error

    #Group them by labels
    unique_labels = np.unique(labels)
    if cmaps is None:
        cmap = plt.cm.get_cmap('tab20', len(unique_labels))
        cmaps = [cmap(i) for i in range(len(unique_labels))]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    any_nan_x = False
    any_nan_y = False
    if insert_inset:
        ax_inset = fig.add_axes([0.4, 0.35, 0.45, 0.45])
        max_x, max_y = 0, 0
        min_x, min_y = 0, 0
    for i, label in enumerate(unique_labels):
        label_err1 = normalized_error1[labels == label]
        label_err2 = normalized_error2[labels == label]
        mean_err1 = np.mean(label_err1)
        mean_err2 = np.mean(label_err2)

        any_nan_x = any_nan_x or np.isnan(label_err1).any()
        any_nan_y = any_nan_y or np.isnan(label_err2).any()

        # Replace NaN values with 1.2
        label_err1 = np.nan_to_num(label_err1, nan=1.2)
        label_err2 = np.nan_to_num(label_err2, nan=1.2)

        if insert_inset:
            max_x = max(max_x, np.max(label_err1))
            min_x = min(min_x, np.min(label_err1))
            max_y = max(max_y, np.max(label_err2))
            min_y = min(min_y, np.min(label_err2))

        ax.scatter(
            label_err1, label_err2, label=label,
            alpha=0.5, color=cmaps[i], s=size)
        ax.scatter(
            mean_err1, mean_err2, label=label,
            alpha=1, color=cmaps[i], s=size)

        if insert_inset:
            ax_inset.scatter(label_err1, label_err2, label=label, alpha=0.5, color=cmaps[i], s=size)
            ax_inset.scatter(mean_err1, mean_err2, label=label, alpha=1, color=cmaps[i], s=size)

    ax.plot([0, 1.2], [0, 1.2], color='black')
    ax.plot([1.0, 1.0], [0, 1.2], color='black', linestyle='dotted')
    ax.plot([0.0, 1.2], [1.0, 1.0], color='black', linestyle='dotted')
    ax.set_xlim(0, 1.2) #  if any_nan_x else 1.0
    ax.set_ylim(0, 1.2) #  if any_nan_y else 1.0
    xticks = ax.get_xticks()
    ax.set_xticklabels([
        'NaN' if np.isclose(label, 1.2, atol=0.1) else f'{label:.1f}'
        for label in xticks])
    yticks = ax.get_yticks()
    ax.set_yticklabels([
        'NaN' if np.isclose(label, 1.2, atol=0.1) else f'{label:.1f}'
        for label in yticks])
    if insert_inset:
        ax_inset.set_xlim(min_x, max_x if not any_nan_x else 1.2)
        ax_inset.set_ylim(min_y, max_y if not any_nan_y else 1.2)
        xticks = ax_inset.get_xticks()
        ax_inset.set_xticklabels([
            'NaN' if np.isclose(label, 1.2, atol=0.1) else f'{label:.1f}'
            for label in xticks])
        yticks = ax_inset.get_yticks()
        ax_inset.set_yticklabels([
            'NaN' if np.isclose(label, 1.2, atol=0.1) else f'{label:.1f}'
            for label in yticks])
        ax_inset.plot([1.0, 1.0], [0, 1.2], color='black', linestyle='dotted')
        # Add background to inset
        renderer = fig.canvas.get_renderer()
        coords = ax.transAxes.inverted().transform(ax_inset.get_tightbbox(renderer))
        border = 0.02
        w, h = coords[1] - coords[0] + 2*border
        ax.add_patch(plt.Rectangle(coords[0] - border, w, h, fc="white",
                                transform=ax.transAxes, zorder=2, ec="red", linewidth=2))
        ax.add_patch(plt.Rectangle(
            np.asarray([min_x, min_y]) - border,
            max_x - min_x + 2 * border,
            max_y - min_y + 2 * border,
            facecolor='none',
            ec="red", linewidth=2))
        ax.plot([min_x - border, coords[0][0] + border], [max_y + border, coords[0][1] + border], color='red', linestyle='dashed')
        ax.plot([max_x + border, 1.17], [max_y + border, coords[0][1] + border], color='red', linestyle='dashed')


    ax.set_xlabel(f'{model_names[0]}')
    ax.set_ylabel(f'{model_names[1]}')
    ax.set_title(title)
    fig.tight_layout()

    if save_to is not None:
        plt.savefig(save_to + '.png')

    return fig

def plot_loss(
    data: List[Dict[str, Union[np.ndarray, str]]],
    parameter: Dict[str, Union[str, np.ndarray]],
    title: str,
    save_to: Optional[str] = None
) -> plt.Figure:
    """
    Plot the loss values for different models over varying parameter values.

    Parameters
    ----------
    data : List[Dict[str, Union[np.ndarray, str]]]
        List of dictionaries, each containing 'losses' (array of shape (n, m)) and 'method' (name of the model).
    parameter : Dict[str, Union[str, np.ndarray]]
        Dictionary with 'name' (name of the parameter) and 'values' (array of parameter values).
    title : str
        Title of the plot.
    save_to : Optional[str], default=None
        Directory path where plots should be saved. If None, no plots will be saved.

    Returns
    -------
    plt.Figure
        The matplotlib figure object containing the plot.
    """
    parameter_values = parameter['values']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = yaml.safe_load(open('style.yaml'))
    for model in data:
        losses = model['losses']
        mean_loss = np.nan_to_num(np.mean(losses, axis=1), nan=0.5)
        std_loss = np.nan_to_num(np.std(losses, axis=1), nan=0.0)

        ax.plot(parameter_values, mean_loss,
                color=colors[model['method']]['dark'], label=model['method'])

        if (std_loss > 0).any():
            ax.fill_between(
                parameter_values, mean_loss - std_loss, mean_loss + std_loss,
                color=colors[model['method']]['light'],
                alpha=0.5)
    ax.set_xlabel(parameter["name"])
    yticks = ax.get_yticks()
    ax.set_yticklabels([
        'NaN' if np.isclose(label, 1.2, atol=0.1) else f'{label:.1f}'
        for label in yticks])
    ax.legend()
    fig.tight_layout()
    plt.title(title)
    if save_to is not None:
        plt.savefig(save_to + '.png')

    return fig

