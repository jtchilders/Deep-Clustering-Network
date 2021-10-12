#!/bin/bash

for BATCH in 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288
do
   echo "Testing batch-size: $BATCH"
   python train_simple_ae.py --data ~whopkins/sigclustering/sigclustering/ccMET_noBackground.h5 --time-throughput --batch-size $BATCH > log_$BATCH.txt
   echo "Done!"
done