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

import wandb
import re
from load_from_wandb import wandb_config 

# Authenticate with W&B (if necessary)
wandb.login()

# Initialize the API
api = wandb.Api()

# Get all runs from the project
runs = api.runs(f"{wandb_config["entity"]}/{wandb_config["project"]}")

split_pattern = re.compile(r"split_(\d+\.?\d*)")
dimension_pattern = re.compile(r"dim_(\d+)")
interaction_pattern = re.compile(r"interaction_(.*?)_dt")
sinkhorn_pattern = re.compile(r"sinkhorn_(\d+\.?\d*)")
for run in runs:
    GROUP_NAME = None
    if "RNA" in run.name:
        GROUP_NAME = "RNA"
    else:
        match = split_pattern.search(run.name)
        if not match:
            continue

        split_ratio = match.group(1)

        if "split_trajectories_True" in run.name:
            GROUP_NAME = "split-trajectories"
        else:
            GROUP_NAME = "split-particles"
        
        GROUP_NAME += f"-ratio_{split_ratio}"

        match = dimension_pattern.search(run.name)
        if match:
            dimension = match.group(1)
            if int(dimension) > 2:
                GROUP_NAME = "ablation-dimension"

        match = sinkhorn_pattern.search(run.name)
        if match:
            sinkhorn = match.group(1)
            if float(sinkhorn) > 1e-12:
                GROUP_NAME = "ablation-couplings"

        match = interaction_pattern.search(run.name)
        if match:
            interaction = match.group(1)
            if interaction != "none":
                GROUP_NAME = "all-energies"
    if GROUP_NAME is None:
        continue
    run.group = GROUP_NAME
    run.update()
    print(f"Updated run {run.name} with group: {GROUP_NAME}")