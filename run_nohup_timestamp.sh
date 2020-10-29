#!bin/bash

DATE_TIME=$(date +\[%m-%d\]%H.%M.%S)
nohup python train.py >> $DATE_TIME.txt &
sleep 1
echo ''
echo 'saving at: '$DATE_TIME'.txt'
