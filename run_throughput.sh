#!/bin/bash

echo "Removing previous logs"
rm log_*.txt

start=`date +%s`

for BATCH in 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288
do
   echo "Measuring with batch-size: $BATCH"
   python train_simple_ae.py --time-throughput --batch-size $BATCH > log_$BATCH.txt
   echo "Done!"
done

end=`date +%s`
runtime=$((end-start))

echo "Runtime: $runtime sec"