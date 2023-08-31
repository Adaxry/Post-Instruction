# Post-Instruction


## Contents

* [Overview](#overview)
* [Requirements](#requirements)
* [Quick to Use](#quick-to-use)
* [Experiments](#experiments)
* [Citation](#citation)
* [Contact](#contact)

## Overview
<p align="center">
  <img src="https://github.com/Adaxry/Post-Instruction/blob/main/figures/examples.png" alt="examples" width="800"/>
</p>


Current instruct-following data generally put the task instruction before the input sentence (referred as "Pre-Ins") for sequqnce generation tasks (e.g., machine translation). We observe that LLMs may forget the frontmost task instruction when the input sentence is long, thus we propose to simply place the task instruction after the input sentence (referred as "Post-Ins"). Both our theoretical and experimental analyses show that Post-Ins pays larger attentions on the model's instruction-following capabilities, yielding consistent performance improvements across two common sequence generation tasks. For more details, please refer to our [technical report](https://arxiv.org/abs/2308.12097) (Instruction Position Matters in Sequence Generation with Large Language Models).

## Requirements
+ transformers>=4.28.0.dev0+
+ python>=3.8.0
+ torch>=1.10
+ deepspeed>=0.8.3+
+ datasets>=2.9.0+


## Quick to Use

+ Organizing original data into Post-Ins format  
  We provide processed training data used in our experiments at [here](https://github.com/Adaxry/Post-Instruction/blob/main/data/README.md), and you can also process your sentence pairs into Post-Ins format by the following script:
  ```bash
  sh scripts/organize_data.sh    # you can replace the file with yours in this script
  ```
  
+ Fine-tuning LLMs
  ```bash
  sh scripts/train_wmt.sh    # take the machine translation as an example
  ```

+ Testing
  ```bash
  sh test/test_wmt.sh   # take the machine translation as an example
  ```


## Experiments
We provide all the model [outputs](https://github.com/Adaxry/Post-Instruction/tree/main/results) for both the machine translation and text summarization task for an easy comparison. Below are partial results of the experiment:

<p align="center">
  <img src="https://github.com/Adaxry/Post-Instruction/blob/main/figures/wmt22.png" alt="wmt" width="800"/>
</p>
<p align="center">
  Results on WMT22 for machine translation. 
</p>

<p align="center">
  <img src="https://github.com/Adaxry/Post-Instruction/blob/main/figures/cnndm.png" alt="wmt" width="400"/>
</p>
<p align="center">
  Results on CNN/DailyMail for long text summarization.
</p>

<p align="center">
  <img src="https://github.com/Adaxry/Post-Instruction/blob/main/figures/heatmap_update.png" alt="wmt" width="900"/>
</p>
<p align="center">
  Pre-Ins mainly foucs on the source input, while Post-Ins pay more attentions on the specific task instruction.
</p>


## Citation
Please cite this paper if you find this repo useful.
```
@article{liu2023instruction,
  title={Instruction Position Matters in Sequence Generation with Large Language Models},
  author={Liu, Yijin and Zeng, Xianfeng and Meng, Fandong and Zhou, Jie},
  journal={arXiv preprint arXiv:2308.12097},
  year={2023}
}
```

## Contact
Please feel free to contact us (yijinliu@tencent.com) for any further questions.
