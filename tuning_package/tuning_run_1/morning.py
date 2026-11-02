"""Tuning run 1 status: every expected run, its exit code, epochs, best validation AP over epochs 0-4.

A run counts as a completed rung result only if it exited 0 and has finite validation AP for epochs 0 to 4.
"""
import csv
import json
import math
import os
import re

os.chdir("/home/jovyan/k194855/tabular_fraud")
EXPECTED = [f"{m}_lr{lr}" for m in ("chosen", "reference", "flat") for lr in ("1e-4", "1e-3")]

starts, ends = {}, {}
if os.path.isfile("runs/tuning/queue.log"):
    for line in open("runs/tuning/queue.log"):
        m = re.match(r"start (\S+) (\S+)", line)
        if m:
            starts[m.group(1)] = m.group(2)
            ends.pop(m.group(1), None)  # a new attempt: the previous attempt's end no longer applies
        m = re.match(r"end (\S+) rc=(-?\d+) (\S+)", line)
        if m:
            ends[m.group(1)] = (int(m.group(2)), m.group(3))


def read_ap(folder):
    path = os.path.join(folder, "training_metrics.csv")
    if not os.path.isfile(path):
        return None
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return {}
    col = next((c for c in rows[0] if c.strip() == "valid_ap"), None)
    ecol = next((c for c in rows[0] if c.strip() == "epoch"), None)
    out = {}
    for i, r in enumerate(rows):
        if col and r.get(col) not in (None, ""):
            epoch = int(float(r[ecol])) if ecol and r.get(ecol) not in (None, "") else i
            out[epoch] = float(r[col])
    return out


def line(name, folder, rc=None, started=None, ended=None, baseline=False):
    ap = read_ap(folder) if folder and os.path.isdir(folder) else None
    manifest = os.path.join(folder or "", "run_manifest_v2.json")
    status_v2 = ""
    if folder and os.path.isfile(manifest):
        try:
            status_v2 = json.load(open(manifest)).get("status", "")
        except Exception:
            status_v2 = "unreadable manifest"
    first5 = [ap.get(e) for e in range(5)] if ap else []
    complete = bool(ap) and all(v is not None and math.isfinite(v) for v in first5)
    if baseline:
        status = "recorded 3e-4 run" if complete else "baseline missing epochs 0-4"
    elif not folder or not os.path.isdir(folder):
        status = "MISSING (never started)" if name not in starts else "FAILED before writing output"
    elif name in starts and name not in ends:
        status = "RUNNING or killed"
    elif name not in ends:
        status = "no end line in queue.log"
    elif rc != 0:
        status = f"FAILED rc={rc}"
    elif not complete:
        status = "INCOMPLETE or non-finite AP"
    else:
        status = "complete"
    took = ""
    if started and ended:
        from datetime import datetime
        f = "%Y-%m-%dT%H:%M:%SZ"
        took = f"{(datetime.strptime(ended, f) - datetime.strptime(started, f)).total_seconds() / 3600:.2f} h"
    best = ""
    if complete and status in ("complete", "recorded 3e-4 run"):
        b = max(first5)
        best = f"best5={b:.4f} at epoch {first5.index(b)}"
    curve = " ".join(f"{ap[e]:.4f}" for e in sorted(ap)) if ap else ""
    print(f"{name:17s} {status:30s} {best:24s} {took:8s} {status_v2:15s} | {curve}")


base = json.load(open("runs/tuning/paths.json")) if os.path.isfile("runs/tuning/paths.json") else {}
for model, role in (("chosen", "C"), ("reference", "R"), ("flat", "F")):
    line(f"{model}_lr3e-4", base.get(role, {}).get("folder"), baseline=True)
    for lr in ("1e-4", "1e-3"):
        name = f"{model}_lr{lr}"
        rc, ended = ends.get(name, (None, None))
        line(name, f"runs/tuning/{name}", rc=rc, started=starts.get(name), ended=ended)
