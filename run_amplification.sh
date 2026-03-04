#!/bin/zsh

cd /Users/sahithbodla/Documents/alphazero
source .venv/bin/activate

# Run with unbuffered Python output
python -u amplification_pipeline.py --config configs/amplify_pos2812.yaml 2>&1 | tee -a amplification_pos2812_run.log
