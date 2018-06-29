#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train \
    --solver=models/partialDAoffice31/dslr_to_amazonC10/protos/solver_dc.prototxt \
    --weights=models/partialDAoffice31/dslr_to_amazonC10/snapshots_dc/pretrained_dc.caffemodel \
    --gpu 0 2>&1 | tee -a ./log/partial_dslr_to_amazonC10.log
