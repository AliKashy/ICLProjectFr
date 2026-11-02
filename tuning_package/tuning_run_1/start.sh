cd /home/jovyan/k194855/tabular_fraud
PY=/home/jovyan/k194855/tabular_fraud_s3_venv_FAIfNI/bin/python
if pgrep -af "python[^ ]* .*(train_v2|train_sequential|score_v2)|bash runs/tuning/lr_screen.sh"; then echo "a job is already running; not starting"; exit 1; fi
$PY runs/tuning/locate.py || exit 1
echo "=== queue started $(date -u +%FT%TZ)" >> runs/tuning/queue.log
nohup bash runs/tuning/lr_screen.sh >> runs/tuning/queue.log 2>&1 &
sleep 5; tail -3 runs/tuning/queue.log
