#!/bin/bash

model=music
export MODEL_NAME=$model
# export CKPT_DIR=./ckpt/
export CKPT_DIR='/home/xu1415/data/randomD/DETrain-public/tf-ckpt'
ENABLE_TRACE=0  ./run example/tensorflow/MusicTransformer-tensorflow2.0/train.py --data_path example/tensorflow/MusicTransformer-tensorflow2.0/dataset/piano
