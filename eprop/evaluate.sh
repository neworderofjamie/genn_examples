#!/bin/bash
for E in {0..49}
do
    python tonic_classifier_evaluate.py --trained-epoch $E "$@"
done
