# config.py
import yaml
from glob import glob

with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

# ────────────────────────────────
#  Back-compat & convenience glue
# ────────────────────────────────
# 1) glob pattern → list
if "h5_paths_glob" in CONFIG and "h5_paths" not in CONFIG:
    CONFIG["h5_paths"] = sorted(glob(CONFIG.pop("h5_paths_glob")))

# 2) old single-file key → list
if "h5_paths" not in CONFIG and "h5_path" in CONFIG:
    CONFIG["h5_paths"] = [CONFIG.pop("h5_path")]

# 3) sanity-check
if "h5_paths" not in CONFIG or len(CONFIG["h5_paths"]) == 0:
    raise ValueError("No HDF5 embedding files found — check your config.")
