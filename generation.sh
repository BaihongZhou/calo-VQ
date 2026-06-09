#!/usr/bin/env bash
# Generate DarkSHINE ECAL showers from a trained stage-2 (CondGPT) run.
# Point --model at the stage-2 log directory (the one holding configs/ and checkpoints/).
PY=/Users/zhoubaihong/miniconda3/envs/caloVQ/bin/python

# Constant 4 GeV incident energy (current single-energy dataset):
$PY gen-tools.py --out darkshine_gen.h5 \
  --model logs/<step2-run-dir> \
  --energy 4000.0 --nevts 10000 --batch-size 256

# Multi-energy: draw incident energies from an existing file's `condition` column:
# $PY gen-tools.py --out darkshine_gen.h5 \
#   --model logs/<step2-run-dir> \
#   --cond-file DarkSHINE_data/export.h5 --nevts 10000 --batch-size 256

echo DONE
