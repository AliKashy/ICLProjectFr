# Tuning run 1: learning-rate screen of the three contenders

CL, 2 November 2026. This is the light learning-rate check of `request_6/CL_DRAFT.md` section 5 step 2, for the owner's three contenders. It applies the owner's rule:

> Let's drop clearly weak settings after 4 or 5 epochs.

The scripts and this sheet include ChatGPT's review of the first version (issue #7, comment 6571480609).

## What runs

Each contender gets learning rates 1e-4 and 1e-3, on either side of its recorded 3e-4 run. Each run is capped at 5 epochs (epochs 0 to 4). The runs go one after another on the one GPU.

Each run starts from the as-run config of a recorded 3e-4 run with the same seed, so the learning rate is the only setting that changes.

| Run folders (`runs/tuning/`) | Model | Starts from the recorded run | Its Test 1 AP |
|---|---|---|---|
| `chosen_lr1e-4`, `chosen_lr1e-3` | `sequential_hier_allpos_h20_chunk`, seed 43 (V2 trainer) | `ALLPOS_H20_CHUNK_seed43` | 0.0912 |
| `reference_lr1e-4`, `reference_lr1e-3` | `sequential_hier_h20`, seed 43 (V2 trainer) | `B_H20_seed43` | 0.0743 |
| `flat_lr1e-4`, `flat_lr1e-3` | `sequential_h15`, seed 42 (V1 flat trainer) | the A-H15 attempt `20261013T044303Z-4bcdf2dd...`, `history-h15/` | 0.0928 |

**Time.** None of these models has a recorded per-epoch training time on a dedicated GPU, so no estimate is given. The queue log records each run's start and end, and the morning report shows each run's hours.

## The scripts (all in `tuning_run_1.zip`)

- **`start.sh`**
  - Refuses to start if a trainer, a scorer or this queue is already running.
  - Runs `locate.py`, then starts `lr_screen.sh` in the background, appending to `runs/tuning/queue.log`.
- **`locate.py`** finds the three recorded runs, wherever the run folders are now.
  - **How it matches:** by their settings (model type, history, seed, chunk layout, learning rate 3e-4), with a `training_metrics.csv` present.
  - **Where it looks:** the whole project. It looks inside `archive/` only if nothing is found outside it.
  - **Copies:** byte-identical copies of a run (config and metrics) count as one run, and it lists them. Two different runs that both match stop it, and it asks which one to use.
  - **Files the runs need:** it checks that every local file the chosen configs read exists, for example the predictor list and `test.yaml`.
  - **Code:** the code must still be at its old paths, `waqas/v2/scripts/train_v2.py` and `waqas/phase4a/scripts/train_sequential.py`. The run folders may have moved; the code may not have.
  - **Output:** it prints the config and metrics file it chose for each run, and writes `runs/tuning/paths.sh` and `runs/tuning/paths.json`.
- **`lr_screen.sh`** is the queue.
  - A run whose folder already exists is skipped, never overwritten. To run one again, rename its folder.
  - Logs are appended, never truncated.
  - Each run's exit code is logged, and the last line counts the runs that exited with an error.
- **`morning.py`** reports on all six expected runs and the three recorded runs.
  - **Statuses:** complete, FAILED (with exit code), FAILED before writing output, INCOMPLETE or non-finite AP, RUNNING or killed, or MISSING.
  - **What it shows for each run:** the best validation AP over epochs 0 to 4, only for runs that are complete; its hours; the V2 manifest status; every epoch's validation AP.

## Commands (pod terminal)

Put `tuning_run_1.zip` in `/home/jovyan/k194855/tabular_fraud`, then:

```bash
cd /home/jovyan/k194855/tabular_fraud
mkdir -p runs/tuning && /home/jovyan/k194855/tabular_fraud_s3_venv_FAIfNI/bin/python -m zipfile -e tuning_run_1.zip runs/tuning/
bash runs/tuning/start.sh
```

It should print the three configs it chose and a `start chosen_lr1e-4` line. If it prints `STOP` or "a job is already running", photograph that and start nothing.

## In the morning

```bash
cd /home/jovyan/k194855/tabular_fraud
cat runs/tuning/queue.log
/home/jovyan/k194855/tabular_fraud_s3_venv_FAIfNI/bin/python runs/tuning/morning.py
```

Photograph both outputs. Do not score any `runs/tuning/` checkpoint yet. The V2 trainer refuses to resume a run whose checkpoint has been scored.

## Decision rule (declared before any result)

- **Rung score:** for each complete run, the best validation AP over epochs 0 to 4. For each recorded 3e-4 run, its first five epochs. Failed, incomplete, non-finite or missing runs are not compared; they are re-run first, or recorded as failed.
- **Drop:** a learning rate whose rung score is more than 0.03 below the best rung score of the same model. This is a resource policy declared in advance, not a significance test. It does not prove that a slower learner could not recover over 10 epochs.
- **Keep the rest, and take them to 10 epochs.**
  - A kept B run continues where it stopped. The V2 trainer allows this for a run whose status is `budget_reached`; a 5-epoch run with patience 5 cannot early-stop, so its status is `budget_reached`. Use the run's own saved config:

    ```bash
    cd /home/jovyan/k194855/tabular_fraud
    PY=/home/jovyan/k194855/tabular_fraud_s3_venv_FAIfNI/bin/python
    nohup $PY -m waqas.v2.scripts.train_v2 --config runs/tuning/<run>/resolved_train_v2.yaml training.max_epochs=10 --resume >> runs/tuning/<run>.log 2>&1 &
    ```

  - The flat trainer cannot resume. A kept flat learning rate gets one new 10-epoch run in its own folder:

    ```bash
    cd /home/jovyan/k194855/tabular_fraud
    PY=/home/jovyan/k194855/tabular_fraud_s3_venv_FAIfNI/bin/python
    source runs/tuning/paths.sh
    nohup $PY waqas/phase4a/scripts/train_sequential.py --config $F training.learning_rate=<rate> training.max_epochs=10 output.checkpoint_dir=runs/tuning/flat_lr<rate>_full >> runs/tuning/flat_lr<rate>_full.log 2>&1 &
    ```

- **Scoring:** after the 10-epoch runs, each model's best learning rate is scored on June SELECT.
- **Comparing models:**
  - Within a model, the learning-rate comparison is paired, because the seed is the same.
  - Across models, the seeds differ (43 for the two B models, 42 for the flat model). So a change to the architecture choice needs the planned finalist comparison at common seeds, and the chosen model stays unless that comparison beats it by a resolved margin.

## After this run

Apply Copilot's reorganization to the pod (`POD_STEPS.md`, not yet written) only after these runs and their resumes have finished. A resume refuses changed code paths, and the reorganization changes them.
