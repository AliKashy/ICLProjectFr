cd /home/jovyan/k194855/tabular_fraud
PY=/home/jovyan/k194855/tabular_fraud_s3_venv_FAIfNI/bin/python
source runs/tuning/paths.sh
failed=0
run() {
  if [ -e "runs/tuning/$1" ]; then echo "skip $1: runs/tuning/$1 already exists (rename it to run it again)"; return; fi
  echo "start $1 $(date -u +%FT%TZ)"
  "${@:2}" training.max_epochs=5 output.checkpoint_dir=runs/tuning/$1 >> "runs/tuning/$1.log" 2>&1
  rc=$?
  [ "$rc" -ne 0 ] && failed=$((failed + 1))
  echo "end $1 rc=$rc $(date -u +%FT%TZ)"
}
for LR in 1e-4 1e-3; do run chosen_lr$LR $PY -m waqas.v2.scripts.train_v2 --config $C training.learning_rate=$LR; done
for LR in 1e-4 1e-3; do run reference_lr$LR $PY -m waqas.v2.scripts.train_v2 --config $R training.learning_rate=$LR; done
for LR in 1e-4 1e-3; do run flat_lr$LR $PY waqas/phase4a/scripts/train_sequential.py --config $F training.learning_rate=$LR; done
echo "queue finished $(date -u +%FT%TZ); runs that exited with an error: $failed"
