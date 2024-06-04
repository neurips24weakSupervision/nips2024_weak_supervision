# nips2024_weak_supervision
This repository contains the code for the publication "Object-Centric Representation Learning with Weakly Supervised Slot Attention," which was submitted to NIPS 2024.

The code is an extended version of Slot Attention's initial implementation: https://github.com/google-research/google-research/tree/master/slot_attention . Make sure to install the specified packages.

# Model Training:
Start the training procedure via: 
python -m nips2024_weak_supervision.object_discovery.train --training_strategy=mask --model_complexity=cnn --weighting_factor=0.1 --model_dir=$checkpoint_path$

Possible options for "training_strategy" are:
-"mask"
-"label"
-"unsupervised"

Possible options for "model_complexity" are:
-"unsupervised"
-"resnet"

Model training requires the Clevrtex dataset to be compiled (see below).

# Inference:
Inference is started with: 
python -m nips2024_weak_supervision.object_discovery.inference --training_strategy=mask --model_complexity=cnn --weighting_factor=0.1 --model_dir=$checkpoint_path$

Possible options for "training_strategy" are:
-"mask"
-"label"
-"unsupervised"

Possible options for "model_complexity" are:
-"unsupervised"
-"resnet"

The inference output is several numpy arrays stored in the $checkpoint_path$. Those are needed for the evaluation in the evaluation/ folder.

# Evaluation:
We evaluate models on their object discovery capability. Additionally, we assess slots for their representation quality of underlying generating factors. Therefore, we bind slots to their underlying objects and run prediction and disentanglement tests. We use a modified version of the disentanglement library: https://github.com/google-research/disentanglement_lib


# Dataset
The code is designed to work on the Clevrtex dataset. Dataset processing is described by Biza et al.:
https://github.com/google-research/google-research/tree/master/invariant_slot_attention/datasets/clevrtex
