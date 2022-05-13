# DETrain

This repository is a solution to **deterministic training**
and **replayable chekcpointing** for DL training programs.
The implementation is based on our paper [Checkpointing and Deterministic Training for Deep Learning](https://www.cs.purdue.edu/homes/taog/docs/CAIN22.pdf).

## Overview

Our solution consists of two parts: a modified version of Tensorflow that supports
deterministic training, and a dynamic analysis system that traces and instruments
programs to support checkpointing and replay.

## System Requirements

We use `python 3.6` to develop the system. 

To build our modified version of 
Tensorflow, users also need the dependency required by Tensorflow.

We recommend our users use a python `virtual environment` to run our system.

To run the example, users need the dependency required by it. We will discuss this part later. 

## Deterministic Tensorflow

We modify `Tensorflow 2.1.0` to support deterministic training. It is maintained
in [this repository](https://github.com/XZ-X/tensorflow-det/tree/detrain).
The related code is on the branch `detrain`.

### Step 1: Download the source code

```
git clone https://github.com/XZ-X/tensorflow-det.git
cd tensorflow-det/
git checkout detrain
```

### Step 2: Build and install Tensorflow

```
# Configure the project following the configure script
./configure

# Build Tensorflow, this step may take 2 hours
bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package

# Build the python package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package <Dir to store the package>

# Uninstall the original version of Tensorflow
pip3 uninstall tensorflow

# Install the deterministic version of Tensorflow
pip3 install <Dir to store the package>/tensorflow-2.1.0-cp36-cp36m-linux_x86_64.whl

```

## Deterministic Training

### Overview

DETrain uses the following structures to model a training program:
```
for epoch in ...:
  ...
  for batch in ...:
    data = dataloader.next_batch()
    ...
    model = update(model, graidents)
```

It then instruments the program as follow:
```python
for epoch in ...:
[+] record_epoch(epoch)
  ...
  for batch in ...:
    ...
    data = dataloader.next_batch()
[+] execute = record_batch(batch)
[+] if not execute:
[+]   continue
    ...
    model = update(model, graidents)
```

As shown in the above code snippet, DETrain records the epoch number for each epoch.
Users can specify the frequency of checkpointing (i.e., making checkpoints for every `n` epoches). The epoch number is saved in the checkpoint files.

When we want to resume the execution from a checkpoint, DETrain leverages the insruments
in the batch loop. If the epoch number is less than the saved epoch number,
DETrain only executes the data loading snippets and skips the loop body.

### Example

Let's go through DETrain's workflow via an example.
This example is a transformer model written in Tensorflow.
Its code and data can be found in `example/tensorflow/MusicTransformer-tensorflow2.0/`.
The entry file is `train.py`.

For now, suppose that we already have the information about its important code structure (discussed above),
as shown in `tf-ckpt/inst.info`.

#### Step1: Build

```
cd syscalls/
make all
```

#### Step2: Modify the scripts

We need to modify two scripts to run the example.
First, change the `<FULL PATH>` in `run-tf-example.sh` to the full path of the `DETrain` directory.
Second, specify the `CUDA_VISIBLE_DEVICES` env variable in `run`.

#### Step3: Specify the checkpoint frequency

At `detrain/tf_handler.py:284`, modify `iter_counter in [3, 6]` to change the frequency
of checkpointing. `iter_counter` is the epoch number. `iter_counter in [3, 6]` means
we save a checkpoint when the epoch number is `3` or `6`.

#### Step4: Run
Run `./run-tf-example.sh` to train the example model with DETrain.

At the first time we run the example, DETrain will automatically make checkpoints
when the target epoch number is reached. We can expect checkpoint files to appear in
`tf-ckpt`. After the checkpoint files appear, when we rerun the script, DETrain
automatically fast-forwards the training program to the latest checkpoint and 
resume the exeuction from that checkpoint. We can rerun the script for multiple times.
The training program should achieve exactly the same accuracy/loss for each run.

### (Optional) Automation

Previously, we explictly tell DETrain the key structure of the training program
via the file `tf-ckpt/inst.info`. Another option is to let DETrain deduce such information via tracing.

DETrain needs to trace the training program twice. In the first round,
it locates the epoch loop and the batch loop. In the second round,
we change the size of the dataset. DETrain then deduces the data loader
by finding variables whose size changes with the size of dataset.

Note that each round of the tracing process could take several hours.

#### Step 1: First round

Change the content of `run-tf-example.sh` to the following command:

```
#!/bin/bash

model=music
export MODEL_NAME=$model
export CKPT_DIR='<FULL PATH>/DETrain-public/tf-ckpt'
ENABLE_TRACE=1  ./run example/tensorflow/MusicTransformer-tensorflow2.0/train.py --data_path example/tensorflow/MusicTransformer-tensorflow2.0/dataset/piano
```

and run this script.

A file named `trace.music` is expected in `DETrain-public/` and a file named
`inst.info` is expected in `DETrain-public/tf-ckpt`. The former contains the 
execution trace. DETrain uses it to deduce the data loader in the second round.
The latter contains the structure information collected in the first round.

#### Step 2: Second round

Change the last command in `run-tf-example.sh` to:

```
ENABLE_TRACE=1 DIFF_SIZE=312 ./run example/tensorflow/MusicTransformer-tensorflow2.0/train.py --data_path example/tensorflow/MusicTransformer-tensorflow2.0/dataset/piano1
```

and run this script.

Note that we change the dataset to `piano1` and we use an env variable to tell
DETrain the change of size should be `312`.

After this step, the file `DETrain-public/tf-ckpt/inst.info` should contain
similar information as the provided one.

### PyTorch example

TBD.


## Contributors

Hongyu Liu, Xiangzhe Xu, Guanhong Tao, Zhou Xuan, Xiangyu Zhang