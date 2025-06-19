#!/usr/bin/bash

docker build -t rl-train -f ./train.dockerfile .
docker run -d --rm \
    -v ./src/checkpoints/policies/:/workspace/checkpoints/policies \
    --name rl-train-4 \
    rl-train \
    python3 train.py --report-interval 50 --history-len 4
docker run -d --rm \
    -v ./src/checkpoints/policies/:/workspace/checkpoints/policies \
    --name rl-train-12 \
    rl-train \
    python3 train.py --report-interval 50 --history-len 12