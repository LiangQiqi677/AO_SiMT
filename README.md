# AO-SiMT
Source code for the paper "Alignment Offset Based Adaptive Training for Simultaneous Machine Translation"

## Contents
- [AO-SiMT](#ao-simt)
  - [Contents](#contents)
  - [Introduction](#introduction)
  - [Usage](#usage)
  - [Requirements](#requirements)

## Introduction
+ Implementation

Implemented based on [Fairseq](https://github.com/pytorch/fairseq), a sequence modeling toolkit that allows researchers and developers to train custom models for translation, summarization, language modeling and other text generation tasks.

+ Data

[WMT15 German-English](http://www.statmt.org/wmt15/translation-task.html)

[NIST Chinese-English](https://www.ldc.upenn.edu/)

## Usage
Note: The usage is on the top of Fairseq, for more details, please refer to the user manual of Fairseq.
+ Calculating AO
```
python AO_calculate.py
```

+ Data Preprocess
```
sh data_preprocess.sh
```

+ Training
  + '+AW'：Only add the AO-based adaptive loss weight to the training objective.
  ```
  sh train_aw.sh
  ```
  + '+CL'：Only add the AO-based curriculum learning training.
  ```
  sh train_cl.sh
  ```
  + '+AW +CL'：add both '+AW' and '+CL'.
  ```
  sh train_aw_cl.sh
  ```

+ Eval
```
sh eval.sh
```

## Requirements
+ Python version \>=3.6
+ PyTorch version \>=1.4.0
