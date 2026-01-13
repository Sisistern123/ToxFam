# config.py
import yaml
import argparse
import sys
from glob import glob

# 1. Parse Command Line Arguments
parser = argparse.ArgumentParser()
# Default to new_combined if no flag is provided
parser.add_argument("--config", type=str, default="model/configs/new_combined.yaml", help="Path to config file")

# Use parse_known_args to avoid crashing if other scripts add their own args later
args, _ = parser.parse_known_args()

print(f"Loading Configuration from: {args.config}")

# 2. Load the YAML file specified by the argument
try:
    with open(args.config, "r") as f:
        CONFIG = yaml.safe_load(f)
except FileNotFoundError:
    print(f"ERROR: Config file not found at {args.config}")
    sys.exit(1)

# 3. Back-compatibility & convenience glue (Same as before)
if "h5_paths_glob" in CONFIG and "h5_paths" not in CONFIG:
    CONFIG["h5_paths"] = sorted(glob(CONFIG.pop("h5_paths_glob")))

if "h5_paths" not in CONFIG and "h5_path" in CONFIG:
    CONFIG["h5_paths"] = [CONFIG.pop("h5_path")]

if "h5_paths" not in CONFIG or len(CONFIG["h5_paths"]) == 0:
    # Optional: check if h5_paths is actually needed for the strategy before raising error,
    # but generally safe to keep this check.
    raise ValueError("No HDF5 embedding files found — check your config.")