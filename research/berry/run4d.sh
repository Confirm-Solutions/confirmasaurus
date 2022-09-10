start=0
end=10
python confirm/confirm/berrylib/batch_run.py \
    --name "berry4d" \
    --n-arms 4 \
    --n-theta-1d 64 \
    --sim-size 500000 \
    --theta-min -3.5 \
    --theta-max 1.0 \
    --gridpt-batch-begin $start \
    --gridpt-batch-end $end