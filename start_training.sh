#!/bin/bash

# Start the training job on 8 GPUs and run it in the background
nohup torchrun --standalone --nproc-per-node=8 trainer.py > output.txt 2>&1 &