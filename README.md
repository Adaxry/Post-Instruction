# Post-Instruction


## Contents

* [Overview](#overview)
* [Quick to Use](#quick-to-use)
* [Experiments](#experiments)
* [Self-Attention HeatMap](#self-attention-heatmap) 
* [Contact](#contact)

## Overview
<p align="center">
  <img src="https://github.com/Adaxry/Post-Instruction/blob/main/figures/examples.png" alt="examples" width="800"/>
</p>


Current instruct-following data generally put the task instruction before the input sentence (referred as "Pre-Ins") for sequqnce generation tasks (e.g., machine translation). We observe that LLMs may forget the frontmost task instruction when the input sentence is long, thus we propose to simply place the task instruction after the input sentence (referred as "Post-Ins"). Both our theoretical and experimental analyses show that Post-Ins pays larger attentions on the model's instruction-following capabilities, yielding consistent performance improvements across two common sequence generation tasks. For more details, please refer to our [technical report](https://arxiv.org/abs/2308.12097) (Instruction Position Matters in Sequence Generation with Large Language Models).


## Quick to Use

+ Organizing original data into Post-Ins format
  ```bash
  sh  scripts/organize_data.sh  # you can replace the file with yours in this script
  ```
  




## Experiments

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

## Self-Attention HeatMap
<p align="center">
  <img src="https://github.com/Adaxry/Post-Instruction/blob/main/figures/heatmap.png" alt="wmt" width="900"/>
</p>
<p align="center">
  Pre-Ins mainly foucs on the source input, while Post-Ins pay more attentions on the specific task instruction.
</p>

## Contact
Please feel free to contact us (yijinliu@tencent.com) for any further questions.
