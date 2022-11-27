#!/bin/sh
while true
do
    echo "Trying to get a TPU VM"
    gcloud compute tpus tpu-vm create tpu4 --zone=europe-west4-a --accelerator-type=v3-8 --version=tpu-vm-pt-1.12
    echo "Sleeping for 5 minutes"
    sleep 300 # wait 5 minutes before trying again
done