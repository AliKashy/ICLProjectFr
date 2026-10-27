# Pod clean-up script (27 October 2026)

[`releases/pod_reorg_20261027_v3.zip`](releases/pod_reorg_20261027_v3.zip) holds one file, `tools/organize_pod.py`. It reorganizes my project folder on the bank's machine: one folder per model, one folder per seed inside it with only the latest results, one shared folder, and one `archive/` for everything superseded. It only renames files on the same disk; it never deletes or changes a file. It finds my package folder by itself, so there is nothing to personalize.

## Run it

1. Download the zip ("Download raw file") and upload it to the project root on the bank's machine.
2. In a terminal there:

```
python -m zipfile -e pod_reorg_20261027_v3.zip .
python tools/organize_pod.py plan
```

The first line creates (or replaces) `tools/organize_pod.py`. The second only reads the folders and writes `shared/reorg/REPORT.txt`; nothing moves.

3. I send `REPORT.txt` to CL and CH. After they confirm it:

```
python tools/organize_pod.py apply
python tools/organize_pod.py verify
```

## The layout after "apply"

```
models/<model>/seed<NN>/   the scored checkpoint, its config and metrics, launch_config.yaml, scores/
shared/                    populations, data look, measurements, comparisons, seed check, logs, code history
tools/                     the hand-run scripts
archive/<original path>    everything superseded; I delete it when satisfied
```

The code and package files stay where they are. `ablations/` and `runs/` end up empty and are removed.

## Safety

- Before its first move, "apply" checks that nothing changed since "plan" (same items, sizes, file counts and file times, nothing new), that no training or scoring job is running, that the report belongs to this plan and lists no conflict, and that every destination is free, on the same disk and writable. If any check fails, nothing moves.
- If "apply" is interrupted, running it again carries on. "plan" refuses to run once moves have started.
- "verify" passes only when every checkpoint and prediction hash matches at its new path, every planned move is logged, and the old folders are gone.
