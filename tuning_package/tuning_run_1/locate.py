"""Find the three recorded learning-rate-3e-4 runs that tuning run 1 starts from, wherever the run folders now are.

The run folders may have moved; the code must still be at its pre-reorganization paths (waqas/v2/scripts/train_v2.py,
waqas/phase4a/scripts/train_sequential.py). The whole project is searched (archive/ only if nothing is found
outside it). Copies whose config and metrics files are byte-identical count as one run; two different matching runs
are a conflict and stop the queue. Every local file the chosen configs read must exist. Writes runs/tuning/paths.sh
and runs/tuning/paths.json.
"""
import hashlib
import json
import math
import os
import sys

import yaml

ROOT = "/home/jovyan/k194855/tabular_fraud"
os.chdir(ROOT)

PREFERRED = {
    "C": ["runs/v2/seed_robustness/ALLPOS_H20_CHUNK_seed43", "models/arch_b_h20_chunk/seed43",
          "models/sequential_hier_allpos_h20_chunk/seed43"],
    "R": ["runs/v2/seed_robustness/B_H20_seed43", "models/arch_b_h20_sliding/seed43",
          "models/sequential_hier_h20/seed43"],
    "F": ["ablations/arch_a_flat/history_h15/20261013T044303Z-4bcdf2ddbdf940dfa8638239d2c791ab/history-h15",
          "models/arch_a_h15/seed42/history-h15", "models/arch_a_h15/seed42", "models/sequential_h15/seed42"],
}
NAMES = {"C": "chosen (all-position h20 chunk, seed 43)", "R": "reference (hierarchical h20, seed 43)",
         "F": "best flat (h15, seed 42)"}
SKIP_DIRS = {".git", "__pycache__", "node_modules", "tuning"}


def get(cfg, dotted):
    for key in dotted.split("."):
        if not isinstance(cfg, dict) or key not in cfg:
            return None
        cfg = cfg[key]
    return cfg


def matches(role, cfg):
    hist, seed, lr = get(cfg, "sequence.history_len"), get(cfg, "training.seed"), get(cfg, "training.learning_rate")
    try:
        if lr is None or not math.isclose(float(lr), 3e-4, rel_tol=1e-9):
            return False
    except (TypeError, ValueError):
        return False
    if role == "C":
        return (get(cfg, "model.type") == "FTTransformerHier" and hist == 20 and seed == 43
                and get(cfg, "sequence.predict_every_transaction") is True
                and get(cfg, "sequence.all_position_layout") == "nonoverlapping_chunks"
                and not get(cfg, "sequence.train_target_ending"))
    if role == "R":
        return (get(cfg, "model.type") == "FTTransformerHier" and hist == 20 and seed == 43
                and not get(cfg, "sequence.predict_every_transaction"))
    return (get(cfg, "model.type") in ("FTTransformerSeq", None) and hist == 15 and seed == 42
            and get(cfg, "sequence.time_delta") is not False
            and get(cfg, "sequence.attention_mask") != "causal")


def digest(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def search(skip_archive):
    hits = {role: [] for role in NAMES}
    for top, dirs, files in os.walk("."):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("tabular_fraud_s3_venv")
                         and not (skip_archive and d == "archive"))
        metrics = os.path.join(top, "training_metrics.csv")
        if not os.path.isfile(metrics):
            continue
        for name in ("resolved_train_v2.yaml", "resolved_train.yaml"):
            config = os.path.join(top, name)
            if not os.path.isfile(config):
                continue
            try:
                with open(config) as fh:
                    cfg = yaml.safe_load(fh)
            except Exception:
                continue
            for role in NAMES:
                if matches(role, cfg):
                    hits[role].append({"folder": os.path.normpath(top), "config": os.path.normpath(config),
                                       "metrics": os.path.normpath(metrics),
                                       "key": digest(config) + digest(metrics)})
            break
    return hits


def rank(hit, role):
    prefs = PREFERRED[role]
    return (prefs.index(hit["folder"]) if hit["folder"] in prefs else len(prefs), len(hit["folder"]), hit["folder"])


outside = search(skip_archive=True)
inside = None
chosen, problems = {}, []
for role in ("C", "R", "F"):
    hits = outside[role]
    where = "outside archive/"
    if not hits:
        inside = inside or search(skip_archive=False)
        hits, where = inside[role], "including archive/"
    groups = {}
    for hit in hits:
        groups.setdefault(hit["key"], []).append(hit)
    if not groups:
        problems.append(f"{NAMES[role]}: no matching run found ({where})")
        continue
    if len(groups) > 1:
        problems.append(f"{NAMES[role]}: {len(groups)} different matching runs ({where}); say which one to use:")
        for copies in groups.values():
            problems.append("    " + ", ".join(h["folder"] for h in copies))
        continue
    copies = sorted(next(iter(groups.values())), key=lambda h: rank(h, role))
    pick = copies[0]
    chosen[role] = {k: pick[k] for k in ("folder", "config", "metrics")}
    print(f"{NAMES[role]}:\n    config  {pick['config']}\n    metrics {pick['metrics']}")
    if len(copies) > 1:
        print("    identical copies also at: " + ", ".join(h["folder"] for h in copies[1:]))

for path in ("waqas/v2/scripts/train_v2.py", "waqas/phase4a/scripts/train_sequential.py"):
    if not os.path.isfile(path):
        problems.append(f"trainer missing: {path} (the code must still be at its old paths)")

SKIP_KEYS = ("output.", "eval.", "experiment.")


def file_refs(cfg, prefix=""):
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            yield from file_refs(v, f"{prefix}{k}.")
    elif isinstance(cfg, str) and not prefix.startswith(SKIP_KEYS) and not cfg.startswith("s3://"):
        if cfg.startswith("/") or cfg.endswith((".csv", ".yaml", ".yml", ".json", ".joblib", ".parquet", ".pt")):
            yield prefix[:-1], cfg


for role, pick in chosen.items():
    with open(pick["config"]) as fh:
        cfg = yaml.safe_load(fh)
    for key, ref in file_refs(cfg):
        if not os.path.exists(ref):
            problems.append(f"{NAMES[role]}: {key} = {ref} does not exist")

if problems:
    print("STOP, send a photo of this:")
    for p in problems:
        print("  " + p)
    sys.exit(1)

os.makedirs("runs/tuning", exist_ok=True)
with open("runs/tuning/paths.json", "w") as fh:
    json.dump(chosen, fh, indent=1)
with open("runs/tuning/paths.sh", "w") as fh:
    for role in ("C", "R", "F"):
        fh.write(f"{role}={chosen[role]['config']}\n")
print("paths written to runs/tuning/paths.sh")
