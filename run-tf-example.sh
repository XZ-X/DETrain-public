#!/bin/bash

model=music
export MODEL_NAME=$model
export CKPT_DIR='<FULL PATH>/DETrain-public/tf-ckpt'
ENABLE_TRACE=0  ./run example/tensorflow/MusicTransformer-tensorflow2.0/train.py --data_path example/tensorflow/MusicTransformer-tensorflow2.0/dataset/piano


# ENABLE_TRACE=1  ./run example/tensorflow/MusicTransformer-tensorflow2.0/train.py --data_path example/tensorflow/MusicTransformer-tensorflow2.0/dataset/piano


# ENABLE_TRACE=1 DIFF_SIZE=312 ./run example/tensorflow/MusicTransformer-tensorflow2.0/train.py --data_path example/tensorflow/MusicTransformer-tensorflow2.0/dataset/piano1